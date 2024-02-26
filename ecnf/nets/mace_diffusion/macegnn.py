from typing import Optional, List
import jax
import jax.numpy as jnp
from flax import linen as nn
import e3nn_jax as e3nn
import jraph
import chex

from ecnf.utils.graph import get_graph_inputs
import ecnf.nets.mace_tools as tools

from .blocks_jax import (
    DiffusionInteractionBlock,
    EquivariantProductBasisBlock,
    NonLinearReadoutBlock,
)

from .utils_jax import get_edge_vectors_and_lengths


class MACEDiffusionAdapted(nn.Module):
    """This implementation tries to modify the diffusion model appropriately."""
    dim: int
    MLP_irreps: e3nn.Irreps
    hidden_irreps: e3nn.Irreps
    num_interactions: int
    num_species: int  # earlier, num_elements
    n_nodes: int
    graph_type: str
    avg_num_neighbors: float
    max_ell: int  # 5 from Laurence's implementation (earlier, 3)
    r_max: float = None  # currently not used

    variance_scaling_init: float = 0.001
    scale_output: bool = False
    normalization_factor: int = 1
    correlation: int = 3
    train_graphs: List[jraph.GraphsTuple] = None  # TODO: get this

    def setup(self):
        assert self.dim in [2, 3]

        if self.avg_num_neighbors == "average":
            assert self.train_graphs is not None
            avg_num_neighbors = tools.compute_avg_num_neighbors(self.train_graphs)
        elif self.avg_num_neighbors is None:
            avg_num_neighbors = self.n_nodes - 1
        else:
            avg_num_neighbors = self.avg_num_neighbors
            
        assert avg_num_neighbors >= 1

        sh_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)
        MLP_irreps = e3nn.Irreps(self.MLP_irreps)
        hidden_irreps = e3nn.Irreps(self.hidden_irreps)

        hidden_dims = hidden_irreps.count(e3nn.Irrep(0, 1))  # n_scalars
        self.embedding = nn.Dense(features=hidden_dims)
        
        mace_layers = []
        for i in range(0, self.num_interactions):
            if i == 0:
                node_feats_irreps = e3nn.Irreps([(hidden_dims, (0, 1))])  # n_scalars x 0e
            else:
                node_feats_irreps = hidden_irreps  # n_scalars x 0e + n_vectors x 1o
            
            mace_layers.append(
                MACE_layer(max_ell=self.max_ell,
                           sh_irreps=sh_irreps,
                           avg_num_neighbors=avg_num_neighbors,
                           correlation=self.correlation,
                           num_species=self.num_species,
                           node_feats_irreps=node_feats_irreps,
                           hidden_irreps=hidden_irreps,
                           MLP_irreps=MLP_irreps,
                           variance_scaling_init=self.variance_scaling_init,
                           )
            )
            
        self.mace_layers = mace_layers

        self.final_scaling = self.param("final_scaling", nn.initializers.ones_init(), ()) if self.scale_output else 1.0
         

    def __call__(self,
                 positions: chex.Array,        # (B, n_nodes, dim) 
                 node_features: chex.Array,    # (B, n_nodes)
                 global_features: chex.Array,  # (B, time_embedding_dim)
    ) -> chex.Array:
        assert positions.ndim in (2, 3)
        vmap = (positions.ndim == 3)
        if vmap:
            return jax.vmap(self.call_single)(positions, node_features, global_features)
        else:
            return self.call_single(positions, node_features, global_features)


    def call_single(self,
                    positions: chex.Array,       # (n_nodes, dim) 
                    node_features: chex.Array,   # (n_nodes,)
                    time_embedding: chex.Array,  # (time_embedding_dim,)
        ):
        chex.assert_rank(positions, 2)
        chex.assert_rank(node_features, 1)
        chex.assert_rank(time_embedding, 1)
        chex.assert_axis_dimension(node_features, 0, self.n_nodes)

        _, edge_index = get_graph_inputs(self.graph_type, positions, self.n_nodes, self.r_max, stack=True)
        shifts = 0
        # get edge vectors
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=edge_index, shifts=shifts)  # (n_edges, dim), (n_edges, 1)

        # concat zeros along the z axis if problem is 2D
        if self.dim == 2:
            n_edges = vectors.shape[0]
            vectors = jnp.concatenate([vectors,
                                       jnp.zeros((n_edges, 1))], axis=1)
            positions = jnp.concatenate([positions,
                                       jnp.zeros((self.n_nodes, 1))], axis=1)
            assert vectors.shape == (n_edges, 3)
            assert positions.shape == (self.n_nodes, 3)
            

        # convert atomic numbers to one-hot
        node_attrs = node_features.copy()
        node_attrs_onehot = jax.nn.one_hot(node_features-1, self.num_species) / self.normalization_factor  # (n_nodes, n_species)
        assert node_attrs_onehot.shape == (self.n_nodes, self.num_species)

        
        # broadcast time_embedding to match node_attrs_onehot
        time_embedding = jnp.tile(time_embedding, (self.n_nodes, 1))  # (n_nodes, time_embedding_dim)
        
        node_attrs_and_time = jnp.concatenate([node_attrs_onehot, time_embedding], axis=-1)  # (n_nodes, n_species + time_embedding_dim)

        lengths_0 = lengths
        positions_0 = positions.copy()
        h = self.embedding(node_attrs_and_time)  # (n_species + time_embedding_dim) => n_hidden_scalars
        node_feats = h

        # Many body interactions
        for mace_layer in self.mace_layers:
            # (n_nodes, n_hidden_scalars), (n_nodes, 3), (n_nodes, hidden_irreps.dim)
            many_body_scalars, many_body_vectors, node_feats = mace_layer(
                vectors=vectors,
                lengths=lengths_0,
                node_feats=node_feats,
                node_attrs=node_attrs,
                edge_feats=lengths,
                edge_index=edge_index)

            positions = positions + many_body_vectors.array
            # Node update
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=positions, edge_index=edge_index, shifts=shifts)

        # Output
        predicted_noise_positions = (positions - positions_0)

        predicted_noise_positions = predicted_noise_positions[:, :self.dim]

        predicted_noise_positions = predicted_noise_positions * self.final_scaling

        return predicted_noise_positions


class MACE_layer(nn.Module):
    max_ell: int
    sh_irreps: e3nn.Irreps
    avg_num_neighbors: float
    correlation: int
    num_species: int
    hidden_irreps: e3nn.Irreps
    node_feats_irreps: e3nn.Irreps
    MLP_irreps: e3nn.Irreps
    variance_scaling_init: float

    def setup(self):
        node_feats_irreps = e3nn.Irreps(self.node_feats_irreps)
        sh_irreps = e3nn.Irreps(self.sh_irreps)
        hidden_irreps = e3nn.Irreps(self.hidden_irreps)
        num_features = hidden_irreps.count(e3nn.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        MLP_irreps = e3nn.Irreps(self.MLP_irreps)
        
        self.interaction = DiffusionInteractionBlock(
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            variance_scaling_init=self.variance_scaling_init,
        )

        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=e3nn.Irreps(self.interaction.target_irreps),
            target_irreps=hidden_irreps,
            correlation=self.correlation,
            num_species=self.num_species,
        )

        self.readout = NonLinearReadoutBlock(
            irreps_in=hidden_irreps, 
            MLP_irreps=MLP_irreps, 
            activation=jax.nn.silu, 
            num_species=num_features,
        )


    def __call__(self, vectors, lengths, node_feats, node_attrs, edge_feats, edge_index):
        vectors_sh = e3nn.spherical_harmonics(input=vectors,
                                              irreps_out=self.sh_irreps,
                                              normalize=True,
                                            )
        node_feats = self.interaction(
            node_feats=node_feats,
            edge_attrs=vectors_sh,
            edge_feats=edge_feats,
            lengths=lengths,
            edge_index=edge_index,
        )
        node_feats = self.product(node_feats=node_feats, node_attrs=node_attrs)  
        node_out = self.readout(node_feats)  # (n_nodes, n_featsx0e + 1o)
        
        return node_out[:, :-3], node_out[:, -3:], node_feats


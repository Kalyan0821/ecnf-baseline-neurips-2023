from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import torch
import e3nn_jax as e3nn
# from e3nn import o3

from ecnf.utils.graph import get_graph_inputs

# from mace.tools.diffusion_tools import remove_mean

from .blocks import (
    DiffusionInteractionBlock,
    EquivariantProductBasisBlock,
    NonLinearReadoutBlock,
)

# from .utils import get_edge_vectors_and_lengths
def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender, receiver = edge_index
    # From ase.neighborlist:
    # D = positions[j]-positions[i]+S.dot(cell)
    # where shifts = S.dot(cell)
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


class MACEDiffusionAdapted(nn.Module):
    """This implementation tries to modify the diffusion model appropriately."""
    dim: int
    MLP_irreps: e3nn.Irreps
    hidden_irreps: e3nn.Irreps
    r_max: float
    num_interactions: int
    num_species: int  # earlier, num_elements
    n_nodes: int
    graph_type: str
    avg_num_neighbors: float
    time_embedding_dim: int

    normalization_factor: int = 1
    max_ell: int = 3
    correlation: int = 3

    def setup(self):
        assert self.dim in [2, 3]
        
        hidden_dims = self.hidden_irreps.count("0e")  # n_scalars
        self.embedding = nn.Dense(features=hidden_dims)
        
        mace_layers = []
        for i in range(0, self.num_interactions):
            if i == 0:
                node_feats_irreps = e3nn.Irreps([(hidden_dims, (0, 1))])  # n_scalars x 0e
            else:
                node_feats_irreps = self.hidden_irreps  # n_scalars x 0e + n_vectors x 1o
            
            mace_layers.append(
                MACE_layer(max_ell=self.max_ell,
                           avg_num_neighbors=self.avg_num_neighbors,
                           correlation=self.correlation,
                           num_species=self.num_species,
                           node_feats_irreps=node_feats_irreps,
                           hidden_irreps=self.hidden_irreps,
                           MLP_irreps=self.MLP_irreps,
                           r_max=self.r_max)
                    )
            
        self.mace_layers = nn.Sequential(mace_layers)

    def __call__(self,
                 positions,        # (B, n_nodes, dim) 
                 node_attrs,       # (B, n_nodes)
                 time_embedding):  # (B, time_embedding_dim)
        
        assert positions.shape == (self.n_nodes, self.dim), f"{positions.shape} != ({self.n_nodes}, {self.dim})"
        assert node_attrs.shape == (self.n_nodes,)

        # Embeddings
        shifts = 0
        _, edge_index = get_graph_inputs(self.graph_type, positions, self.n_nodes, self.r_max, stack=True)

        # convert atomic numbers to one-hot
        node_attrs = jax.nn.one_hot(node_attrs-1, self.num_species) / self.normalization_factor   # (n_nodes, n_species)
        assert node_attrs.shape == (self.n_nodes, self.num_species)

        # TODO: broadcast time_embedding to match node_attrs
        assert node_attrs.shape[:-1] == time_embedding.shape[:-1], f"{node_attrs.shape[:-1]} != {time_embedding.shape[:-1]}"
        
        node_attrs_and_time = jnp.concatenate([node_attrs, time_embedding], dim=-1)  # (n_nodes, n_species + time_embedding_dim)
        assert node_attrs_and_time.shape == (self.n_nodes, self.num_species + self.time_embedding_dim)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=edge_index, shifts=shifts)  # edge_vectors

        lengths_0 = lengths
        positions_0 = positions
        h = self.embedding(node_attrs_and_time)  # (n_species + time_embedding_dim) => n_hidden_scalars
        node_feats = h

        # Many body interactions
        for mace_layer in self.mace_layers:
            many_body_scalars, many_body_vectors, node_feats = mace_layer(
                vectors=vectors,
                lengths=lengths_0,
                node_feats=node_feats,
                edge_feats=lengths,
                edge_index=edge_index)

            positions = positions + many_body_vectors
            # Node update
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=positions, edge_index=edge_index, shifts=shifts)

        # Output
        # predicted_noise_positions = remove_mean(positions, data.batch) - positions_0
        predicted_noise_positions = positions - positions_0
        assert predicted_noise_positions.shape == (self.n_nodes, self.dim)

        return predicted_noise_positions


class MACE_layer(nn.Module):
    max_ell: int
    avg_num_neighbors: float
    correlation: int
    num_species: int
    hidden_irreps: e3nn.Irreps
    node_feats_irreps: e3nn.Irreps
    MLP_irreps: e3nn.Irreps
    r_max: Optional[float] = None

    def setup(self):
        node_attr_irreps = e3nn.Irreps([(self.num_species, (0, 1))])
        num_features = self.hidden_irreps.count(e3nn.Irrep(0, 1))
        edge_feats_irreps = e3nn.Irreps([(num_features, (0, 1))])
        sh_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        # self.spherical_harmonics = o3.SphericalHarmonics(
        #     sh_irreps, normalize=True, normalization="component"
        # )
        self.spherical_harmonics = lambda x: e3nn.spherical_harmonics(irreps_out=sh_irreps,
                                                                      input=x,
                                                                      normalize=True,
                                                                      normalization="component")

        self.interaction = DiffusionInteractionBlock(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=self.node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=self.hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            r_max=self.r_max,
        )
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=self.interaction.target_irreps,
            target_irreps=self.hidden_irreps,
            correlation=self.correlation,
            element_dependent=False,
            num_species=self.num_species,
            use_sc=False,
        )
        self.readout = NonLinearReadoutBlock(
            self.hidden_irreps, self.MLP_irreps, nn.activation.silu, num_features
        )


    def __call__(self, vectors, lengths, node_feats, edge_feats, edge_index):

        edge_attrs = self.spherical_harmonics(vectors)
        node_feats, sc = self.interaction(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            lengths=lengths,
            edge_index=edge_index,
        )
        node_feats = self.product(node_feats=node_feats, sc=sc, node_attrs=None)
        node_out = self.readout(node_feats).squeeze(-1)  # (n_nodes, n_feats x 0e + 1 x 1o)
        return node_out[:, :-3], node_out[:, -3:], node_feats


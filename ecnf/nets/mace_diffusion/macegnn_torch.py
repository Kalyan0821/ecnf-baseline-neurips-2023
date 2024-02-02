from typing import Any, Callable, Dict, Optional
import jax
import torch
from e3nn import o3

from ecnf.utils.graph import get_graph_inputs

# from mace.tools.diffusion_tools import remove_mean

from .blocks import (
    DiffusionInteractionBlock,
    EquivariantProductBasisBlock,
    NonLinearReadoutBlock,
)
from .utils import get_edge_vectors_and_lengths


class MACEDiffusionAdapted(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 MLP_irreps: o3.Irreps,
                 hidden_irreps: o3.Irreps,
                 r_max: float,
                 num_interactions: int,
                 num_species: int,  # earlier, num_elements
                 n_nodes: int,
                 graph_type: str,
                 avg_num_neighbors: float,
                 time_embedding_dim: int,

                 normalization_factor: int,
                 max_ell: int = 3,
                 correlation: int = 3
    ):   
        super().__init__()

        assert dim in [2, 3]

        hidden_dims = hidden_irreps.count(o3.Irrep(0, 1))  # n_hidden_scalars
        n_dims_in = num_species + time_embedding_dim
        self.embedding = torch.nn.Linear(n_dims_in, hidden_dims)
        
        self.dim = dim
        self.num_species = num_species
        self.r_max = r_max
        self.n_nodes = n_nodes
        self.graph_type = graph_type
        self.avg_num_neighbors = avg_num_neighbors
        self.normalization_factor = normalization_factor

        self.node_mlps = torch.nn.ModuleList([])
        self.mace_layers = torch.nn.ModuleList([])
        for i in range(0, num_interactions):
            self.node_mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dims * 2, hidden_dims),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dims, hidden_dims),
                )
            )
            if i == 0:
                node_feats_irreps = o3.Irreps(
                    [(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))]
                )
            else:
                node_feats_irreps = hidden_irreps
            self.mace_layers.append(
                MACE_layer(
                    max_ell=max_ell,
                    avg_num_neighbors=avg_num_neighbors,
                    correlation=correlation,
                    num_species=num_species,
                    node_feats_irreps=node_feats_irreps,
                    hidden_irreps=hidden_irreps,
                    MLP_irreps=MLP_irreps,
                    r_max=r_max
                )
            )

    def forward(self,
                positions,        # (B, n_nodes, dim) 
                node_attrs,       # (B, n_nodes)
                time_embedding):  # (B, time_embedding_dim)
        
        assert positions.shape == (self.n_nodes, self.dim)
        assert node_attrs.shape == (self.n_nodes,)

        # Embeddings
        shifts = 0
        _, edge_index = get_graph_inputs(self.graph_type, positions, self.n_nodes, self.r_max, stack=True)

        # convert atomic numbers to one-hot
        node_attrs = jax.nn.one_hot(node_attrs-1, self.num_species) / self.normalization_factor   # (n_nodes, n_species)
        assert node_attrs.shape == (self.n_nodes, self.num_species)

        # TODO: broadcast time_embedding to match node_attrs
        assert node_attrs.shape[:-1] == time_embedding.shape[:-1], f"{node_attrs.shape[:-1]} != {time_embedding.shape[:-1]}"
        
        node_attrs_and_time = torch.cat([node_attrs, time_embedding], dim=-1)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=edge_index, shifts=shifts)  # edge_vectors

        lengths_0 = lengths
        positions_0 = positions
        h = self.embedding(node_attrs_and_time)  # (n_species + time_embedding_dim) => n_hidden_scalars
        node_feats = h

        # Many body interactions
        for node_mlp, mace_layer in zip(self.node_mlps, self.mace_layers):
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


class MACE_layer(torch.nn.Module):
    def __init__(
        self,
        max_ell: int,
        avg_num_neighbors: float,
        correlation: int,
        num_species: int,
        hidden_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        interaction_cls: Optional[Callable] = DiffusionInteractionBlock,
        r_max: Optional[float] = None,
    ):
        super().__init__()
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        edge_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        self.interaction = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            r_max=r_max,
        )
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=self.interaction.target_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation,
            element_dependent=False,
            num_species=num_species,
            use_sc=False,
        )
        self.readout = NonLinearReadoutBlock(
            hidden_irreps, MLP_irreps, torch.nn.SiLU(), num_features
        )

    def forward(
        self, vectors, lengths, node_feats, edge_feats, edge_index
    ):
        edge_attrs = self.spherical_harmonics(vectors)
        node_feats, sc = self.interaction(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            lengths=lengths,
            edge_index=edge_index,
        )
        node_feats = self.product(node_feats=node_feats, sc=sc, node_attrs=None)
        node_out = self.readout(node_feats).squeeze(-1)  # (n_nodes, n_featsx0e + 1x1o)
        return node_out[:, :-3], node_out[:, -3:], node_feats


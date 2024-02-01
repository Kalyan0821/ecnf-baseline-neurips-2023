###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the ASL License (see ASL.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
from e3nn import o3
import jraph

from ecnf.utils.graph import get_graph_inputs

from mace.data import AtomicData
from mace.data.neighborhood import get_neighborhood
from mace.tools.diffusion_tools import remove_mean
from mace.tools.scatter import scatter_sum
from e3nn import nn as enn

from .blocks import (
    DiffusionInteractionBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    LocalDiffusionInteractionBlock,
    NoiseScheduleBlock,
    NonLinearNodeEmbeddingBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    TerminationInteractionBlock,
)
from .utils import (
    add_noise_position_and_attr,
    add_noise_position_and_attr_local,
    compute_forces,
    dropout_node,
    generate_residual_mask,
    get_central_heavy_atom,
    get_edge_vectors_and_lengths,
    get_noisy_batch,
    get_num_heavy_atoms,
    get_sigma_and_alpha_given_s,
    index_add,
    prepare_batch,
    reconstruct_neighorhood,
    sample_time,
    SNR_weight,
)


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
        
        self.num_species = num_species
        self.r_max = r_max
        self.n_nodes = n_nodes
        self.graph_type = graph_type
        self.avg_num_neighbors = avg_num_neighbors
        self.normalization_factor = normalization_factor

        self.node_mlp = torch.nn.ModuleList([])
        self.mace_layers = torch.nn.ModuleList([])
        for i in range(0, num_interactions):
            self.node_mlp.append(
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

    def forward(
        self,
        positions,
        node_attrs,
        time_embedding):

        node_attrs /= self.normalization_factor  # (num_species,): one-hot?

        # Embeddings
        shifts = 0
        _, edge_index = get_graph_inputs(self.graph_type, positions, self.n_nodes, self.r_max, 
                                         stack=True)
        assert edge_index.shape[0] == 2
        assert node_attrs.shape == time_embedding.shape, f"{node_attrs.shape} != {time_embedding.shape}"

        node_attrs_and_time = torch.cat([node_attrs, time_embedding], dim=-1)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=edge_index, shifts=shifts)  # edge_vectors

        lengths_0 = lengths
        positions_0 = positions
        h = self.embedding(node_attrs_and_time)  # (num_species + time_embedding_dim) => n_hidden_scalars
        node_feats = h

        # Many body interactions
        for node_mlp, mace_layer in zip(self.node_mlp, self.mace_layers):
            many_body_scalars, many_body_vectors, node_feats = mace_layer(
                vectors=vectors,
                lengths=lengths_0,
                node_feats=node_feats,
                edge_feats=lengths,
                edge_index=edge_index)

            positions = positions + many_body_vectors
            # Node update
            h = h + node_mlp(torch.cat([h, many_body_scalars], dim=-1))
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=positions, edge_index=edge_index, shifts=shifts)

        # Output
        predicted_noise_positions = remove_mean(positions, data.batch) - positions_0
        
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
    ) -> Dict[str, Any]:
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


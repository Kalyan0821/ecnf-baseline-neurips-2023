###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the ASL License (see ASL.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
from e3nn import o3

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


class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        max_steps: int,
        noise_increments: np.array,
        gate: Optional[Callable],
    ):
        super().__init__()
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers
        self.max_steps = max_steps
        self.noise_scheduler = NoiseScheduleBlock(
            alphas=noise_increments, timesteps=max_steps
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=o3.Irreps([(num_elements + 1, (0, 1))]),
            irreps_out=node_feats_irreps,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readout

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            element_dependent=True,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps, num_elements))

        for i in range(num_interactions - 1):
            hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                element_dependent=False,
                num_elements=num_elements,
                use_sc=False,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, num_elements
                    )
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps, num_elements))

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Time generation
        t, t_is_zero = sample_time(
            max_steps=self.max_steps,
            training=training,
            num_batch=data.num_graphs,
            include_zero=False,
            device=data.positions.device,
        )
        data.node_attrs = data.node_attrs / 4.0  # Normalize atomic numbers
        (positions, node_attrs, eps) = add_noise_position_and_attr(
            positions=data.positions,
            node_attrs=data.node_attrs,
            t=t,
            noise_scheduler=self.noise_scheduler,
            batch=data.batch,
        )

        # Embeddings
        node_attrs_and_time = torch.cat([node_attrs, t[data.batch]], dim=-1)
        node_feats = self.node_embedding(node_attrs_and_time)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        predicted_noise_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            node_noise = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            predicted_noise_list.append(node_noise)

        # Predicted noise
        predicted_noise_stack = torch.stack(
            predicted_noise_list, dim=1
        )  # [n_nodes, n_steps, n_dims]
        predicted_noise = predicted_noise_stack.sum(dim=1)  # [n_nodes, ]

        predicted_noise_positions = predicted_noise[:, -3:]
        predicted_noise_labels = predicted_noise[:, :-3]
        err_positions = (predicted_noise_positions - eps[:, -3:]).pow(2)
        err_labels = (predicted_noise_labels - eps[:, :-3]).pow(2)
        loss_positions = (
            scatter_sum(err_positions, data.batch, dim=0).sum(dim=-1).squeeze()
        )
        loss_labels = scatter_sum(err_labels, data.batch, dim=0).sum(dim=-1).squeeze()
        num_nodes = data.ptr[1:] - data.ptr[:-1]
        loss = (
            0.5
            * (loss_positions + loss_labels)
            / (num_nodes * (3 + (node_attrs.shape[-1])))
        )
        print("t", t.squeeze())
        print("loss", loss)
        return {
            "predicted_noise_labels": predicted_noise_labels,
            "predicted_noise_positions": predicted_noise_positions,
            "noise_labels": eps[:, :-3],
            "noise_positions": eps[:, -3:],
            "noise_scheduler": self.noise_scheduler,
            "t_is_zero": t_is_zero.squeeze(),
            "loss": loss,
        }


class EGNN(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        n_dims_in: int,
        hidden_dims: int,
        avg_num_neighbors: int,
        max_steps: int,
        noise_increments: np.ndarray,
        normalization_factor: int,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Linear(n_dims_in, hidden_dims)
        self.output = torch.nn.Linear(hidden_dims, n_dims_in)
        self.max_steps = max_steps
        self.noise_scheduler = NoiseScheduleBlock(
            alphas=noise_increments, timesteps=max_steps
        )

        self.avg_num_neighbors = avg_num_neighbors
        self.edge_mlp = torch.nn.ModuleList([])
        self.node_mlp = torch.nn.ModuleList([])
        self.edge_inference_mlp = torch.nn.ModuleList([])
        self.positions_mlp = torch.nn.ModuleList([])
        self.normalization_factor = normalization_factor

        for _ in range(0, num_layers):
            self.edge_mlp.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dims * 2 + 2, hidden_dims),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dims, hidden_dims),
                    torch.nn.SiLU(),
                )
            )
            self.edge_inference_mlp.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dims, 1, bias=False),
                    torch.nn.Sigmoid(),
                )
            )
            self.node_mlp.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dims * 2, hidden_dims),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dims, hidden_dims),
                )
            )
            layer = torch.nn.Linear(hidden_dims, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            self.positions_mlp.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dims * 2 + 2, hidden_dims),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dims, hidden_dims),
                    torch.nn.SiLU(),
                    layer,
                )
            )

    def forward(
        self,
        data: AtomicData,
        training=False,
        generation=False,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        data.positions.requires_grad = True

        if not generation:
            data.positions.requires_grad = True
            # Time generation
            t, t_is_zero = sample_time(
                max_steps=self.max_steps,
                training=training,
                num_batch=data.num_graphs,
                include_zero=False,
                device=data.positions.device,
            )
            data.node_attrs = (
                data.node_attrs / self.normalization_factor
            )  # Normalize atomic numbers
            (positions, node_attrs, eps) = add_noise_position_and_attr(
                positions=data.positions,
                node_attrs=data.node_attrs,
                t=t,
                noise_scheduler=self.noise_scheduler,
                batch=data.batch,
            )
        else:
            data.node_attrs = data.node_attrs / self.normalization_factor
            node_attrs = data.node_attrs
            positions = data.positions

        # Embeddings
        node_attrs_and_time = torch.cat([node_attrs, t[data.batch]], dim=-1)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=data.edge_index, shifts=data.shifts
        )
        lengths_0 = lengths
        positions_0 = positions
        h = self.embedding(node_attrs_and_time)

        sender, receiver = data.edge_index
        for edge_mlp, edge_inference_mlp, node_mlp, positions_mlp in zip(
            self.edge_mlp, self.edge_inference_mlp, self.node_mlp, self.positions_mlp
        ):
            # Edge MLP
            m_ij = edge_mlp(
                torch.cat(
                    [
                        h[sender],
                        h[receiver],
                        lengths,
                        lengths_0,
                    ],
                    dim=-1,
                )
            )
            # Edge inference
            sc = edge_inference_mlp(m_ij)
            # Scattering
            message = (
                scatter_sum(sc * m_ij, receiver, dim=0, dim_size=positions.shape[0])
                / self.avg_num_neighbors
            )
            # Position update
            positions_update_ij = vectors * positions_mlp(
                torch.cat(
                    [
                        h[sender],
                        h[receiver],
                        lengths,
                        lengths_0,
                    ],
                    dim=-1,
                )
            )
            positions = (
                positions
                + scatter_sum(
                    positions_update_ij, receiver, dim=0, dim_size=positions.shape[0]
                )
                / self.avg_num_neighbors
            )
            # Node update
            h = h + node_mlp(torch.cat([h, message], dim=-1))
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=positions, edge_index=data.edge_index, shifts=data.shifts
            )

        # Output
        predicted_noise = self.output(h)
        predicted_noise_positions = remove_mean(positions, data.batch) - positions_0
        predicted_noise_labels = predicted_noise[:, :-1]
        if not generation:
            err_positions = (predicted_noise_positions - eps[:, -3:]).pow(2)
            err_labels = (predicted_noise_labels - eps[:, :-3]).pow(2)
            loss_positions = (
                scatter_sum(err_positions, data.batch, dim=0).sum(dim=-1).squeeze()
            )
            loss_labels = (
                scatter_sum(err_labels, data.batch, dim=0).sum(dim=-1).squeeze()
            )
            num_nodes = data.ptr[1:] - data.ptr[:-1]
            loss = (
                0.5
                * (loss_positions + loss_labels)
                / (num_nodes.max() * (3 + (node_attrs.shape[-1])))
            )
            print("t", t.squeeze())
            print("loss", loss)
            return {
                "predicted_noise_labels": predicted_noise_labels,
                "predicted_noise_positions": predicted_noise_positions,
                "noise_labels": eps[:, :-3],
                "noise_positions": eps[:, -3:],
                "noise_scheduler": self.noise_scheduler,
                "t_is_zero": t_is_zero.squeeze(),
                "loss": loss,
            }
        else:
            return {
                "predicted_noise_labels": predicted_noise_labels,
                "predicted_noise_positions": predicted_noise_positions,
                "noise_scheduler": self.noise_scheduler,
            }


class MACE_layer(torch.nn.Module):
    def __init__(
        self,
        max_ell: int,
        avg_num_neighbors: float,
        correlation: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        interaction_cls: Optional[Callable] = DiffusionInteractionBlock,
        r_max: Optional[float] = None,
    ):
        super().__init__()
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
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
            num_elements=num_elements,
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
        node_out = self.readout(node_feats).squeeze(-1)  # [n_nodes, ]
        return node_out[:, :-3], node_out[:, -3:], node_feats


class DiffusionMACE(torch.nn.Module):
    def __init__(
        self,
        max_ell: int,
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        correlation: int,
        max_steps: int,
        noise_increments: np.array,
        normalization_factor: int,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_irreps.count(o3.Irrep(0, 1))
        n_dims_in = num_elements + 1
        self.num_elements = num_elements
        self.embedding = torch.nn.Linear(n_dims_in, hidden_dims)
        self.output = torch.nn.Linear(hidden_dims, n_dims_in)
        self.max_steps = max_steps
        self.noise_scheduler = NoiseScheduleBlock(
            alphas=noise_increments, timesteps=max_steps
        )

        self.avg_num_neighbors = avg_num_neighbors
        self.node_mlp = torch.nn.ModuleList([])
        self.mace_layers = torch.nn.ModuleList([])
        self.normalization_factor = normalization_factor

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
                    num_elements=num_elements,
                    node_feats_irreps=node_feats_irreps,
                    hidden_irreps=hidden_irreps,
                    MLP_irreps=MLP_irreps,
                )
            )

    def forward(
        self,
        data: AtomicData,
        training=False,
        generation=False,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        if not generation:
            data.positions.requires_grad = True
            # Time generation
            t, t_is_zero = sample_time(
                max_steps=self.max_steps,
                training=training,
                num_batch=data.num_graphs,
                include_zero=False,
                device=data.positions.device,
            )
            data.node_attrs = (
                data.node_attrs / self.normalization_factor
            )  # Normalize atomic numbers
            (positions, node_attrs, eps) = add_noise_position_and_attr(
                positions=data.positions,
                node_attrs=data.node_attrs,
                t=t,
                noise_scheduler=self.noise_scheduler,
                batch=data.batch,
            )
        else:
            data.node_attrs = data.node_attrs / self.normalization_factor
            node_attrs = data.node_attrs
            positions = data.positions

        # Embeddings
        node_attrs_and_time = torch.cat([node_attrs, t[data.batch]], dim=-1)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=data.edge_index, shifts=data.shifts
        )
        lengths_0 = lengths
        positions_0 = positions
        h = self.embedding(node_attrs_and_time)
        node_feats = h

        for node_mlp, mace_layer in zip(
            self.node_mlp,
            self.mace_layers,
        ):
            # Many body interactions
            many_body_scalars, many_body_vectors, node_feats = mace_layer(
                vectors=vectors,
                lengths=lengths_0,
                node_feats=node_feats,
                edge_feats=lengths,
                edge_index=data.edge_index,
            )
            # Edge MLP
            # m_ij = edge_mlp(
            #     torch.cat([h[sender], h[receiver], lengths, lengths_0,], dim=-1,)
            # )
            # # Edge inference
            # sc = edge_inference_mlp(m_ij)
            # # Scattering
            # message = (
            #     scatter_sum(sc * m_ij, receiver, dim=0, dim_size=positions.shape[0])
            #     / self.avg_num_neighbors
            # )
            positions = positions + many_body_vectors
            # Node update
            h = h + node_mlp(torch.cat([h, many_body_scalars], dim=-1))
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=positions, edge_index=data.edge_index, shifts=data.shifts
            )

        # Output
        predicted_noise = self.output(h)
        predicted_noise_positions = remove_mean(positions, data.batch) - positions_0
        predicted_noise_labels = predicted_noise[:, :-1]
        if not generation:
            err_positions = (predicted_noise_positions - eps[:, -3:]).pow(2)
            err_labels = (predicted_noise_labels - eps[:, :-3]).pow(2)
            loss_positions = (
                scatter_sum(err_positions, data.batch, dim=0).sum(dim=-1).squeeze()
            )
            loss_labels = (
                scatter_sum(err_labels, data.batch, dim=0).sum(dim=-1).squeeze()
            )
            num_nodes = data.ptr[1:] - data.ptr[:-1]
            loss = (
                0.5
                * (loss_positions + loss_labels)
                / (num_nodes.max() * (3 + (node_attrs.shape[-1])))
            )
            print("t", t.squeeze())
            print("loss", loss)
            return {
                "predicted_noise_labels": predicted_noise_labels,
                "predicted_noise_positions": predicted_noise_positions,
                "noise_labels": eps[:, :-3],
                "noise_positions": eps[:, -3:],
                "noise_scheduler": self.noise_scheduler,
                "t_is_zero": t_is_zero.squeeze(),
                "loss": loss,
            }
        else:
            return {
                "predicted_noise_labels": predicted_noise_labels,
                "predicted_noise_positions": predicted_noise_positions,
                "noise_scheduler": self.noise_scheduler,
            }


class LocalDiffusionMACE(torch.nn.Module):
    def __init__(
        self,
        max_ell: int,
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        correlation: int,
        max_steps: int,
        r_max: int,
        r_max_nn: int,
        noise_increments: np.array,
        normalization_factor: int,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_irreps.count(o3.Irrep(0, 1))
        n_dims_in = num_elements + 1
        self.num_elements = num_elements
        self.r_max = r_max
        self.r_max_nn = r_max_nn
        self.embedding = torch.nn.Linear(n_dims_in, hidden_dims)
        self.output = torch.nn.Linear(hidden_dims, n_dims_in)
        self.max_steps = max_steps
        self.noise_scheduler = NoiseScheduleBlock(
            alphas=noise_increments, timesteps=max_steps
        )

        self.avg_num_neighbors = avg_num_neighbors
        self.node_mlp = torch.nn.ModuleList([])
        self.mace_layers = torch.nn.ModuleList([])
        self.normalization_factor = normalization_factor

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
                    num_elements=num_elements,
                    node_feats_irreps=node_feats_irreps,
                    hidden_irreps=hidden_irreps,
                    MLP_irreps=MLP_irreps,
                    interaction_cls=LocalDiffusionInteractionBlock,
                    r_max=r_max,
                )
            )

    def forward(
        self,
        data: AtomicData,
        training=False,
        generation=False,
        central_atoms: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        mask_env: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        if not generation:
            (
                positions,
                node_attrs,
                eps,
                mask_env,
                mask_diffuse_env,
                data,
                t,
                t_is_zero,
            ) = prepare_batch(
                data=data,
                r_max_nn=self.r_max_nn,
                r_max=self.r_max,
                normalization_factor=self.normalization_factor,
                noise_scheduler=self.noise_scheduler,
                max_steps=self.max_steps,
                training=training,
            )

        else:
            data.node_attrs = data.node_attrs / self.normalization_factor
            node_attrs = data.node_attrs
            positions = data.positions

        # Embeddings
        node_attrs_and_time = torch.cat([node_attrs, t[data.batch]], dim=-1)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=data.edge_index, shifts=data.shifts
        )
        lengths_0 = lengths
        positions_0 = positions
        h = self.embedding(node_attrs_and_time)
        node_feats = h

        for node_mlp, mace_layer in zip(
            self.node_mlp,
            self.mace_layers,
        ):
            # Many body interactions
            edge_feats = torch.cat(
                [lengths.unsqueeze(0), t[data.batch][data.edge_index[0]].unsqueeze(0)],
                dim=0,
            )
            many_body_scalars, many_body_vectors, node_feats = mace_layer(
                vectors=vectors,
                lengths=lengths_0,
                node_feats=node_feats,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            positions = index_add(positions, many_body_vectors, mask_env, generation)
            # Node update
            h = h + node_mlp(torch.cat([h, many_body_scalars], dim=-1))
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=positions, edge_index=data.edge_index, shifts=data.shifts
            )

        # Output
        predicted_noise_positions = positions - positions_0
        predicted_noise_labels = self.output(h)[:, :-1]
        if not generation:
            predicted_noise_labels_out = torch.zeros_like(predicted_noise_labels)
            predicted_noise_positions_out = torch.zeros_like(predicted_noise_positions)
            predicted_noise_labels_out[mask_env] += predicted_noise_labels[mask_env]
            predicted_noise_positions_out[mask_env] += predicted_noise_positions[
                mask_env
            ]
            print("predicted_noise_positions_out", predicted_noise_positions_out)
        else:
            predicted_noise_labels_out = (
                predicted_noise_labels * mask_env.float().unsqueeze(-1)
            )
            predicted_noise_positions_out = (
                predicted_noise_positions * mask_env.float().unsqueeze(-1)
            )
        if not generation:
            err_positions = (predicted_noise_positions_out[mask_env] - eps[:, -3:]).pow(
                2
            )
            err_labels = (predicted_noise_labels_out[mask_env] - eps[:, :-3]).pow(2)
            loss_positions = (
                scatter_sum(
                    err_positions, data.batch[mask_env], dim=0, dim_size=data.num_graphs
                )
                .sum(dim=-1)
                .squeeze()
            )
            loss_labels = (
                scatter_sum(
                    err_labels, data.batch[mask_env], dim=0, dim_size=data.num_graphs
                )
                .sum(dim=-1)
                .squeeze()
            )
            num_nodes_per_graph = scatter_sum(
                mask_diffuse_env.float(), data.batch, dim=0, dim_size=data.num_graphs
            )
            loss = (
                0.5
                * (loss_positions + loss_labels)
                / (num_nodes_per_graph * (3 + (node_attrs.shape[-1])))
            )
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            return {
                "predicted_noise_labels": predicted_noise_labels_out,
                "predicted_noise_positions": predicted_noise_positions_out,
                "noise_labels": eps[:, :-3],
                "noise_positions": eps[:, -3:],
                "noise_scheduler": self.noise_scheduler,
                "t_is_zero": t_is_zero.squeeze(),
                "loss": loss,
            }
        else:
            return {
                "predicted_noise_labels": predicted_noise_labels_out,
                "predicted_noise_positions": predicted_noise_positions_out,
                "noise_scheduler": self.noise_scheduler,
            }

            # t, t_is_zero = sample_time(
            #     max_steps=self.max_steps,
            #     training=training,
            #     num_batch=data.num_graphs,
            #     include_zero=False,
            #     device=data.positions.device,
            # )
            # p = torch.rand(1)
            # num_heavy_atoms = get_num_heavy_atoms(
            #     node_attrs=data.node_attrs,
            #     batch=data.batch,
            #     num_graphs=data.num_graphs,
            # )
            # n_atom_min = torch.min(num_heavy_atoms).int().item()
            # if p < 0.9 and n_atom_min > 2:
            #     print("walk")
            #     (
            #         data,
            #         central_atoms,
            #         mask_diffuse_env,
            #         patience_trigger,
            #     ) = generate_residual_mask(data)
            #     if mask_diffuse_env is not None:
            #         mask_diffuse_env = mask_diffuse_env.nonzero().squeeze(1)
            #     if mask_diffuse_env is not None and len(mask_diffuse_env) == 0:
            #         print("bulk")
            #         central_atoms, mask_diffuse_env, patience_trigger = None, None, True
            #     if mask_diffuse_env is None:
            #         print("ok")
            #         central_atoms, mask_diffuse_env, patience_trigger = None, None, True
            # else:
            #     print("bulk")
            #     central_atoms, mask_diffuse_env, patience_trigger = None, None, True
            # print("patience", patience_trigger)
            # data.node_attrs = (
            #     data.node_attrs / self.normalization_factor
            # )  # Normalize atomic numbers
            # (positions, node_attrs, eps, mask_env) = add_noise_position_and_attr_local(
            #     positions=data.positions,
            #     node_attrs=data.node_attrs,
            #     edge_index=data.edge_index,
            #     num_graphs=data.num_graphs,
            #     normalization_factor=self.normalization_factor,
            #     t=t,
            #     noise_scheduler=self.noise_scheduler,
            #     patience_trigger=patience_trigger,
            #     batch=data.batch,
            #     central_atom=central_atoms,
            #     mask_env=mask_diffuse_env,
            # )

            # mask_diffuse_env = torch.zeros(
            #     positions.shape[0], dtype=torch.bool, device=positions.device
            # )
            # mask_diffuse_env[mask_env] = True
            # edge_index, shifts = reconstruct_neighorhood(
            #     positions=positions,
            #     ptr=data.ptr,
            #     pbc=data.pbc,
            #     cell=data.cell,
            #     num_graphs=data.num_graphs,
            #     r_max_nn=max(self.r_max_nn, self.r_max),  # TODO: change to r_max_nn
            # )

            # data.edge_index = edge_index
            # data.shifts = shifts

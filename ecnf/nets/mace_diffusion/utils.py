###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the ASL License (see ASL.md)
###########################################################################################

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn
import torch.utils.data
from mace.data.neighborhood import get_neighborhood

from mace.tools import to_numpy
from mace.tools.diffusion_tools import remove_mean
from mace.tools.scatter import scatter_sum
from mace.data import AtomicData
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_min_max
from mace.tools.torch_geometric.utils import maybe_num_nodes, subgraph

from .blocks import AtomicEnergiesBlock


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training=True
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=energy,  # [n_graphs, ]
        inputs=positions,  # [n_nodes, 3]
        grad_outputs=torch.ones_like(energy),
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        only_inputs=True,  # Diff only w.r.t. inputs
        allow_unused=True,
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        logging.warning("Gradient is None, padded with zeros")
        return torch.zeros_like(positions)
    return -1 * gradient


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


def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()

    return mean, std


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    return mean, rms


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []

    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


def sample_time(
    max_steps: int,
    training: bool = True,
    num_batch: int = 1,
    include_zero: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if include_zero:
        t_0 = 0
    else:
        t_0 = 1
    if training:
        times_int = (
            torch.randint(
                t_0,
                max_steps + 1,
                size=(num_batch, 1),
                dtype=torch.get_default_dtype(),
                device=device,
            )
            / max_steps
        )

        time_is_zero = (times_int == 0).float()
        return times_int, time_is_zero
    else:
        times_int = (
            torch.randint(
                t_0,
                max_steps + 1,
                size=(num_batch, 1),
                dtype=torch.get_default_dtype(),
                device=device,
            )
            / max_steps
        )
        time_is_zero = (times_int == 0).float()
        return times_int, time_is_zero


def add_noise_position_and_attr(
    positions: torch.Tensor,
    node_attrs: torch.Tensor,
    batch: torch.Tensor,
    t: int = 0,
    noise_scheduler=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if noise_scheduler is None:
        return positions, node_attrs
    else:
        gamma = noise_scheduler(t)
        alpha = torch.sqrt(torch.sigmoid(-gamma[batch]))
        sigma = torch.sqrt(torch.sigmoid(gamma[batch]))
        positions = remove_mean(positions, batch=batch)
        eps_positions = remove_mean(
            torch.randn_like(positions, dtype=torch.get_default_dtype()), batch=batch
        )
        positions_t = alpha * positions + sigma * eps_positions
        eps_node_attrs = torch.randn_like(node_attrs, dtype=torch.get_default_dtype())
        node_attrs_t = alpha * node_attrs + sigma * eps_node_attrs
    eps = torch.cat([eps_node_attrs, eps_positions], dim=-1)
    return positions_t, node_attrs_t, eps


def add_noise_position_and_attr_local(
    positions: torch.Tensor,
    node_attrs: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    num_graphs: int,
    normalization_factor: float,
    patience_trigger: bool,
    t: int = 0,
    p: float = 0.2,
    noise_scheduler=None,
    central_atom=None,
    mask_env=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if noise_scheduler is None:
        return positions, node_attrs
    else:
        gamma = noise_scheduler(t)
        alpha = torch.sqrt(torch.sigmoid(-gamma[batch]))
        sigma = torch.sqrt(torch.sigmoid(gamma[batch]))
        if central_atom is None:
            central_atom = get_central_heavy_atom(
                node_attrs=node_attrs * normalization_factor,
                batch=batch,
                num_graphs=num_graphs,
            )
        # print("positions", positions.shape)
        positions = positions - positions[central_atom][batch]
        # print("positions_after_sum", positions.shape)
        if patience_trigger:
            mask_env = edge_index[1][torch.isin(edge_index[0], central_atom)]
            prob = torch.rand(mask_env.shape[0], device=edge_index.device)
            prob_mask = (prob > p) + (node_attrs[mask_env, 0] != 0)
            mask_env = mask_env[prob_mask]
        alpha_masked = torch.ones_like(alpha)
        alpha_masked[mask_env] = alpha[mask_env]
        # print("alpha_masked", alpha_masked.shape)
        # eps_positions = remove_mean(
        #     torch.randn_like(positions[mask_env]),
        #     batch=batch[mask_env],
        # )
        eps_positions = 0.4 * torch.randn_like(positions[mask_env])
        # print("eps_positions", eps_positions.shape)
        # print("eps_positions", sigma[mask_env] * eps_positions)
        positions_t = torch.index_add(
            alpha_masked * positions, 0, mask_env, sigma[mask_env] * eps_positions
        )
        eps_node_attrs = torch.randn_like(node_attrs[mask_env])
        node_attrs_t = torch.index_add(
            alpha_masked * node_attrs, 0, mask_env, sigma[mask_env] * eps_node_attrs
        )
    eps = torch.cat([eps_node_attrs, eps_positions], dim=-1)
    return positions_t, node_attrs_t, eps, mask_env


def get_central_heavy_atom(
    node_attrs: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: torch.Tensor,
) -> torch.Tensor:
    """Get the central heavy atom of each graph."""
    if node_attrs.shape[1] == 1:
        central_atom = torch.randint(
            0, node_attrs.shape[0], size=(num_graphs,), device=node_attrs.device
        )
        return central_atom
    mask = torch.zeros_like(node_attrs)
    node_attrs = torch.nn.functional.one_hot(
        torch.argmax(node_attrs, dim=-1), node_attrs.shape[-1]
    )
    mask[:, 0] = node_attrs[:, 0]
    heavy_elements = (node_attrs - mask).sum(dim=1)
    num_heavy_elements = scatter_sum(heavy_elements, batch, dim=0, dim_size=num_graphs)
    mask_int = np.random.randint(
        low=torch.zeros_like(num_heavy_elements).cpu().numpy(),
        high=num_heavy_elements.cpu().numpy(),
    )
    mask_int = torch.tensor(mask_int).long()
    mask_int = torch.cat([mask_int, torch.tensor([-1])]).to(node_attrs.device)
    ind = (
        torch.cat(
            [
                torch.tensor([0], device=node_attrs.device),
                torch.cumsum(num_heavy_elements, dim=0).long(),
            ]
        )
        + mask_int
    )
    return torch.gather(torch.where(heavy_elements > 0)[0], 0, ind[:-1])


def get_num_heavy_atoms(
    node_attrs: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: torch.Tensor,
):
    if node_attrs.shape[1] == 1:
        return scatter_sum(node_attrs, batch, dim=0, dim_size=num_graphs)
    mask = torch.zeros_like(node_attrs)
    node_attrs = torch.nn.functional.one_hot(
        torch.argmax(node_attrs, dim=-1), node_attrs.shape[-1]
    )
    mask[:, 0] = node_attrs[:, 0]
    heavy_elements = (node_attrs - mask).sum(dim=1)
    num_heavy_elements = scatter_sum(heavy_elements, batch, dim=0, dim_size=num_graphs)
    return num_heavy_elements


def SNR_weight(
    t: torch.Tensor, noise_scheduler: Callable, max_steps: int
) -> torch.Tensor:
    """
    Compute the SNR weight for the current time step.

    Args:
        t: current time step
        noise_scheduler: noise scheduler
        max_steps: maximum number of time steps

    Returns:
        SNR weight

    """
    gamma_t = noise_scheduler(t)
    gamma_s = noise_scheduler(t - 1 / max_steps)
    return torch.exp(gamma_t - gamma_s) - 1


def get_sigma_and_alpha_given_s(
    gamma_t: torch.Tensor, gamma_s: torch.Tensor, batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the sigma and alpha given the gamma at time t and time s.

    Args:
        gamma_t: The gamma at time t.
        gamma_s: The gamma at time s.
        batch: The batch vector.
    """
    sigma2_t_given_s = -torch.expm1(
        torch.nn.functional.softplus(gamma_s[batch])
        - torch.nn.functional.softplus(gamma_t[batch])
    )
    log_alpha2_t = torch.nn.functional.logsigmoid(-gamma_t)
    log_alpha2_s = torch.nn.functional.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = alpha_t_given_s[batch]

    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


def get_noisy_batch(
    n_nodes: torch.Tensor,
    num_elements: int,
    cutoff: int,
    pbc: Optional[Tuple[bool]] = (False, False, False),
    cell=None,
    device=None,
) -> "AtomicData":
    """Get a batch of noisy graphs

    Args:
        n_nodes: number of nodes in each graph
        num_elements: number of elements in the dataset
        device: device to store the data
    Returns:
        AtomicData: batch of noisy graphs
    """

    atomic_datas = []
    for n_node in n_nodes:
        positions = 0.4 * torch.randn(n_node, 3)
        node_attrs = torch.randn(n_node, num_elements)
        if cell is None:
            max_postion = (
                np.max(np.absolute(positions.detach().cpu().numpy()), axis=0)
                + 10 * cutoff
            )
            cell = max_postion * np.identity(3, dtype=float)
        edge_index, shifts = get_neighborhood(
            positions=positions.detach().cpu().numpy(),
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
        )
        edge_index_nn, _ = get_neighborhood(
            positions=positions.detach().cpu().numpy(),
            cutoff=1.7,
            pbc=pbc,
            cell=cell,
        )
        forces = torch.zeros_like(positions)
        energy = None
        weight = None
        atomic_data = AtomicData(
            positions=positions,
            node_attrs=node_attrs,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_index_nn=torch.tensor(edge_index_nn, dtype=torch.long),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
            pbc=torch.tensor(pbc, dtype=torch.bool),
            forces=forces,
            energy=energy,
            weight=weight,
        )
        atomic_datas.append(atomic_data)

    batch_loader = torch_geometric.dataloader.DataLoader(
        dataset=atomic_datas,
        batch_size=len(n_nodes),
        shuffle=False,
        drop_last=False,
    )
    return next(iter(batch_loader)).to(device)


def append_noisy_local_environment(
    batch: "AtomicData",
    n_nodes: torch.Tensor,
    central_atoms: torch.Tensor,
    num_elements: int,
    cutoff: int,
    device=None,
):
    """Append a noisy local environment to a batch"""
    noisy_batch = get_noisy_batch(
        n_nodes=n_nodes,
        num_elements=num_elements,
        cutoff=cutoff,
        pbc=tuple(batch.pbc.cpu().numpy())[:3],
        device=device,
    )
    positions = noisy_batch.positions
    batch.positions = batch.positions - batch.positions[central_atoms][batch.batch]
    central_distance = torch.linalg.norm(
        positions - batch.positions[central_atoms][noisy_batch.batch], dim=-1
    )
    rescaling = torch.max(torch.ones_like(central_distance), central_distance / cutoff)
    positions = positions / rescaling[:, None]
    batch.positions = torch.cat([batch.positions, positions], dim=0)
    batch.node_attrs = torch.cat([batch.node_attrs, noisy_batch.node_attrs], dim=0)
    ptr_scaffold = batch.ptr
    batch.ptr = batch.ptr + noisy_batch.ptr

    edge_indices = []
    n_atoms = batch.ptr[1:] - batch.ptr[:-1]
    n_atoms = torch.cat(
        [torch.tensor([0], device=batch.positions.device), n_atoms], dim=0
    )

    n_graph_sum = torch.cumsum(n_atoms, dim=0)
    shifts_list = []
    batch_list = []
    node_mask_list = []
    for n_graph in range(1, len(n_nodes) + 1):
        edge_index, shifts = get_neighborhood(
            positions=batch.positions[n_graph_sum[n_graph - 1] : n_graph_sum[n_graph]]
            .cpu()
            .numpy(),
            cutoff=cutoff,
            pbc=tuple(batch.pbc.cpu().numpy())[3 * (n_graph - 1) : 3 * n_graph],
            cell=batch.cell.cpu().numpy()[
                3 * (n_graph - 1) : 3 * n_graph, 3 * (n_graph - 1) : 3 * n_graph
            ],
        )
        node_mask = torch.zeros(n_graph_sum[n_graph]).to(device)
        node_mask[ptr_scaffold[n_graph] : n_graph_sum[n_graph]] = 1
        node_mask_list.append(node_mask.long())
        edge_indices.append(edge_index + n_graph_sum[n_graph - 1].item())
        shifts_list.append(shifts)
        batch_list.append(
            torch.ones(n_graph_sum[n_graph].item(), dtype=torch.long) * n_graph - 1
        )
    edge_index = torch.tensor(np.concatenate(edge_indices), dtype=torch.long).to(device)
    batch.edge_index = edge_index
    shifts = torch.tensor(np.concatenate(shifts_list), dtype=torch.float).to(device)
    batch.shifts = shifts
    node_mask = torch.cat(node_mask_list, dim=0)
    batch.batch = torch.cat(batch_list, dim=0).to(device)
    batch.cell = noisy_batch.cell
    batch.num_nodes = batch.positions.shape[0]
    return batch, node_mask


def reconstruct_neighorhood(
    positions: torch.Tensor,
    ptr: torch.Tensor,
    pbc: torch.Tensor,
    cell: torch.Tensor,
    num_graphs: int,
    r_max_nn: float,
):
    device = positions.device
    n_atoms = ptr[1:] - ptr[:-1]
    n_atoms = torch.cat([torch.tensor([0], device=device), n_atoms], dim=0)

    n_graph_sum = torch.cumsum(n_atoms, dim=0)

    edge_indices = []
    shifts_list = []
    for n_graph in range(1, num_graphs + 1):
        edge_index, shifts = get_neighborhood(
            positions=positions[n_graph_sum[n_graph - 1] : n_graph_sum[n_graph]]
            .detach()
            .cpu()
            .numpy(),
            cutoff=r_max_nn,
            pbc=tuple(pbc.cpu().numpy())[3 * (n_graph - 1) : 3 * n_graph],
            cell=cell[3 * (n_graph - 1) : 3 * n_graph].cpu().numpy(),
        )
        edge_indices.append(edge_index + n_graph_sum[n_graph - 1].item())
        shifts_list.append(shifts)
    edge_index = torch.tensor(
        np.concatenate(edge_indices, axis=1), dtype=torch.long
    ).to(device)
    shifts = torch.tensor(np.concatenate(shifts_list, axis=0), dtype=torch.float).to(
        device
    )
    return edge_index, shifts


def dropout_node(
    edge_index: torch.Tensor,
    p: float = 0.5,
    num_nodes: Optional[int] = None,
    mask_env: Optional[torch.Tensor] = None,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {p}")

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    if mask_env is not None:
        node_mask = node_mask + mask_env
    edge_index, _, edge_mask = subgraph(
        node_mask, edge_index, num_nodes=num_nodes, return_edge_mask=True
    )
    return edge_index, edge_mask, node_mask


def get_residual_from_graph(
    node_attrs: torch.Tensor,
    edge_index: torch.Tensor,
    central_atoms: torch.Tensor,
    num_nodes: Optional[int] = None,
    p: Optional[float] = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r""" """
    # print("central atoms residual", central_atoms)
    mask_env = ~(
        scatter_sum(
            src=torch.isin(edge_index[0], central_atoms).float(),
            index=edge_index[1],
            dim=0,
            dim_size=num_nodes,
        )
        > 0
    )
    prob = torch.rand(mask_env.shape[0], device=edge_index.device)
    prob_mask = (prob > p) & (node_attrs[:, 0] == 0)
    mask_env = mask_env + prob_mask
    node_mask = torch.index_add(
        mask_env, 0, central_atoms, torch.ones_like(central_atoms) == 1
    )
    edge_index, _, edge_mask, node_idx = subgraph(
        node_mask,
        edge_index,
        num_nodes=num_nodes,
        return_edge_mask=True,
        relabel_nodes=True,
        return_node_idx=True,
    )
    return edge_index, edge_mask, node_mask, node_idx


def get_residual_batch(
    batch: "AtomicData",
) -> "AtomicData":
    r""" """
    central_atoms = get_central_heavy_atom(
        node_attrs=batch.node_attrs, batch=batch.batch, num_graphs=batch.num_graphs
    )
    edge_index, edge_mask, node_mask, node_idx = get_residual_from_graph(
        node_attrs=batch.node_attrs,
        edge_index=batch.edge_index,
        num_nodes=batch.num_nodes,
        central_atoms=central_atoms,
    )
    residual_batch = batch.clone()
    residual_batch.edge_index = edge_index
    residual_batch.positions = batch.positions[node_mask, :]
    residual_batch.node_attrs = batch.node_attrs[node_mask, :]
    residual_batch.batch = batch.batch[node_mask]
    residual_batch.positions = (
        residual_batch.positions - batch.positions[central_atoms][residual_batch.batch]
    )
    residual_batch.num_nodes = len(residual_batch.batch)
    num_nodes_per_graph = scatter_sum(
        torch.ones_like(residual_batch.batch), residual_batch.batch, dim=0
    )
    residual_batch.ptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=batch.positions.device),
            num_nodes_per_graph.cumsum(dim=0),
        ]
    )
    residual_batch.shifts = batch.shifts[edge_mask, :]
    return residual_batch, node_idx


def get_NN_residual_from_graph(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
    num_nodes: Optional[int] = None,
    return_node_idx: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r""" """
    if return_node_idx:
        edge_index, _, edge_mask, node_idx = subgraph(
            node_mask,
            edge_index,
            num_nodes=num_nodes,
            return_edge_mask=True,
            relabel_nodes=True,
            return_node_idx=True,
        )
        return edge_index, edge_mask, node_mask, node_idx
    else:
        edge_index, _, edge_mask = subgraph(
            node_mask,
            edge_index,
            num_nodes=num_nodes,
            return_edge_mask=True,
            relabel_nodes=True,
        )
        return edge_index, edge_mask, node_mask


def get_NN_residual_batch(
    batch: "AtomicData",
    central_atoms: torch.Tensor,
    node_mask: torch.Tensor,
) -> "AtomicData":
    r""" """
    central_distance = torch.linalg.norm(
        batch.positions - batch.positions[central_atoms][batch.batch],
        dim=-1,
    )
    # print("central distance", central_distance)
    nn_mask = central_distance < 10.0
    # print("nn mask", nn_mask)
    # print("node_mask", node_mask)
    node_nn_mask = node_mask * nn_mask + ~node_mask
    # is_carbon_mask = batch.node_attrs[:, 1] == 1
    node_nn_mask = node_nn_mask  # + is_carbon_mask
    dist = torch.linalg.norm((batch.positions - batch.positions.unsqueeze(1)), dim=-1)
    # print("dist", dist)
    dist[dist == 0] = 10000
    H_H_NN_mask = (batch.node_attrs[torch.min(dist, dim=1).indices, 0] == 1).long() & (
        batch.node_attrs[:, 0] == 1
    ).long()
    H_too_far_mask = (torch.min(dist, dim=1).values >= 1.3).long() & (
        batch.node_attrs[:, 0] == 1
    ).long()
    # H_not_nn = (central_distance > 2.1).long() & (batch.node_attrs[:, 0] == 1).long()
    H_filter_mask = H_H_NN_mask.bool() + H_too_far_mask.bool()
    lonely_atom_mask = torch.min(dist, dim=1).values >= 2.0
    # delete hydrogen alone
    node_nn_mask = node_nn_mask * ~H_filter_mask * ~lonely_atom_mask

    node_nn_mask = torch.index_add(
        node_nn_mask.bool(), 0, central_atoms, torch.ones_like(central_atoms) == 1
    )
    # print("node nn mask", node_nn_mask)
    edge_index, edge_mask, node_mask = get_NN_residual_from_graph(
        edge_index=batch.edge_index,
        num_nodes=batch.num_nodes,
        node_mask=node_nn_mask,
    )
    # print("node_idx", node_idx)
    residual_batch = batch.clone()
    residual_batch.edge_index = edge_index
    residual_batch.positions = batch.positions[node_mask, :]
    residual_batch.node_attrs = batch.node_attrs[node_mask, :]
    residual_batch.batch = batch.batch[node_mask]
    residual_batch.positions = (
        residual_batch.positions - batch.positions[central_atoms][residual_batch.batch]
    )
    residual_batch.num_nodes = len(residual_batch.batch)
    num_nodes_per_graph = scatter_sum(
        torch.ones_like(residual_batch.batch), residual_batch.batch, dim=0
    )
    residual_batch.ptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=batch.positions.device),
            num_nodes_per_graph.cumsum(dim=0),
        ]
    )
    residual_batch.shifts = batch.shifts[edge_mask, :]
    return residual_batch


def random_graph_walk(
    node_attrs: torch.Tensor,
    edge_index: torch.Tensor,
    central_atoms: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: torch.Tensor,
    walk_length: int = 2,
):
    # print("central atoms", central_atoms)
    central_atoms_list = [central_atoms]
    ind_env_list = []
    for walk_it in range(walk_length):
        ind_env = edge_index[1][torch.isin(edge_index[0], central_atoms)]
        # print(f"ind env{walk_it}", ind_env)
        if walk_it != 0:
            mask_previous_central = torch.isin(ind_env, central_atoms_walk)
            # print(f"central_atoms_walk{walk_it}", central_atoms_walk)
            ind_env = ind_env[~mask_previous_central]
            # print(f"ind env deleted {walk_it}", ind_env)
            num_heavy_atoms = get_num_heavy_atoms(
                node_attrs=node_attrs[ind_env],
                batch=batch[ind_env],
                num_graphs=num_graphs,
            )
            # print("num heavy atoms", num_heavy_atoms)
            if len(ind_env) == 0 or (num_heavy_atoms == 0).all():
                break
            ind_env_list.append(ind_env)
        ind_central_atoms = get_central_heavy_atom(
            node_attrs=node_attrs[ind_env],
            batch=batch[ind_env],
            num_graphs=num_graphs,
        )
        central_atoms = ind_env[ind_central_atoms]
        if walk_it == 0:
            ind_env_list.append(central_atoms)
        # print("central atoms", central_atoms)
        central_atoms_list.append(central_atoms)
        central_atoms_walk = torch.unique(torch.cat(central_atoms_list))
        print("central atoms walk", central_atoms_walk)
    return torch.unique(torch.cat(ind_env_list))


def generate_residual_mask(
    batch: "AtomicData",
) -> torch.Tensor:
    r""" """
    print("batch", batch)
    device = batch.positions.device
    central_atoms = get_central_heavy_atom(
        node_attrs=batch.node_attrs,
        batch=batch.batch,
        num_graphs=batch.num_graphs,
    )
    print("central atoms", central_atoms)
    central_atoms_bool = torch.zeros_like(batch.batch, dtype=torch.bool)
    central_atoms_bool[central_atoms] = True
    # print("central atoms", central_atoms)
    num_heavy_atoms = get_num_heavy_atoms(
        node_attrs=batch.node_attrs,
        batch=batch.batch,
        num_graphs=batch.num_graphs,
    )
    print("num heavy atoms", num_heavy_atoms)
    n_atom_min = torch.min(num_heavy_atoms).int().item()
    n_atom_min = max(2, n_atom_min - 2)
    print("n atom min", n_atom_min)
    walk_length = torch.randint(1, n_atom_min, (1,)).item()
    patience = 0
    patience_trigger = False
    while patience < 30:
        try:
            mask_walk_env = random_graph_walk(
                node_attrs=batch.node_attrs,
                edge_index=batch.edge_index_nn,
                central_atoms=central_atoms,
                batch=batch.batch,
                num_graphs=batch.num_graphs,
                walk_length=walk_length,
            )
            break
        except:
            print("failed!")
            patience += 1
            walk_length = torch.randint(1, n_atom_min, (1,)).item()
            print("walk length", walk_length)
            # print("new walk length", walk_length)
            pass
    # print("mask walk env", mask_walk_env)
    print("walk length", walk_length)
    print("patience", patience)
    if patience == 30:
        mask_walk_env = torch.zeros_like(batch.batch, dtype=torch.bool)
        mask_walk_env[central_atoms] = True
        patience_trigger = True
        print("patience walk")
        return batch, central_atoms, None, patience_trigger
    mask_walk_bool = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
    mask_walk_bool[mask_walk_env] = True
    mask_diffuse_env_bool = torch.zeros(
        batch.num_nodes, dtype=torch.bool, device=device
    )
    mask_periodic = ~(batch.shifts != 0).sum(1, keepdim=True).bool().squeeze(1)
    mask_diffuse_env = batch.edge_index[1][
        torch.isin(batch.edge_index[0], central_atoms) * mask_periodic
    ]
    # print("mask central local", mask_diffuse_env.nonzero().squeeze(1))
    mask_diffuse_env_bool[mask_diffuse_env] = True
    mask_diffuse_env = mask_diffuse_env_bool * ~mask_walk_bool
    mask_env = mask_walk_bool + mask_diffuse_env
    mask_env = torch.index_add(
        mask_env.bool(),
        0,
        central_atoms,
        torch.ones_like(central_atoms, dtype=torch.bool, device=device),
    )
    edge_index, edge_mask, node_mask = get_NN_residual_from_graph(
        edge_index=batch.edge_index,
        num_nodes=batch.num_nodes,
        node_mask=mask_env,
    )
    positions = batch.positions[node_mask, :]
    dist = torch.linalg.norm(
        (positions[edge_index[1]] - positions[edge_index[0]]),
        dim=-1,
    )
    min_dist = scatter_min_max(dist, edge_index[1])
    # print("min dist", min_dist)
    H_too_far_mask = (min_dist >= 1.3).long() & (
        batch.node_attrs[node_mask, 0] == 1
    ).long()
    # print("H too far mask", H_too_far_mask)
    H_filter_mask = H_too_far_mask.bool()
    lonely_atom_mask = min_dist >= 2.0
    mask_env = ~H_filter_mask * ~lonely_atom_mask

    edge_index_HH, edge_mask_HH, node_mask_HH = get_NN_residual_from_graph(
        edge_index=edge_index,
        num_nodes=len(node_mask),
        node_mask=mask_env,
    )

    residual_batch = batch.clone()
    if node_mask_HH.shape == torch.Size([0]):
        node_mask_HH = torch.arange(
            batch.positions[node_mask, :].shape[0], device=device
        )
        patience_trigger = True
        print("patience H")
        return batch, central_atoms, None, patience_trigger

    residual_batch.edge_index = edge_index_HH
    residual_batch.positions = batch.positions[node_mask, :][node_mask_HH, :]
    residual_batch.node_attrs = batch.node_attrs[node_mask, :][node_mask_HH, :]
    residual_batch.batch = batch.batch[node_mask][node_mask_HH]
    residual_batch.positions = (
        residual_batch.positions - batch.positions[central_atoms][residual_batch.batch]
    )
    residual_batch.num_nodes = len(residual_batch.batch)
    num_nodes_per_graph = scatter_sum(
        torch.ones_like(residual_batch.batch), residual_batch.batch, dim=0
    )
    residual_batch.ptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=batch.positions.device),
            num_nodes_per_graph.cumsum(dim=0),
        ]
    )
    residual_batch.shifts = batch.shifts[edge_mask, :][edge_mask_HH, :]
    central_atoms = central_atoms_bool[node_mask][node_mask_HH].nonzero().squeeze(1)
    return (
        residual_batch,
        central_atoms,
        mask_diffuse_env[node_mask][node_mask_HH],
        patience_trigger,
    )


def prepare_batch(
    data: AtomicData,
    r_max_nn: int,
    r_max: int,
    normalization_factor: float,
    noise_scheduler: Callable,
    max_steps: int,
    training: bool,
):
    # Time generation
    t, t_is_zero = sample_time(
        max_steps=max_steps,
        training=training,
        num_batch=data.num_graphs,
        include_zero=False,
        device=data.positions.device,
    )
    p = torch.rand(1)
    num_heavy_atoms = get_num_heavy_atoms(
        node_attrs=data.node_attrs,
        batch=data.batch,
        num_graphs=data.num_graphs,
    )
    n_atom_min = torch.min(num_heavy_atoms).int().item()
    if p < 0.95 and n_atom_min > 2:
        print("walk")
        (
            data,
            central_atoms,
            mask_diffuse_env,
            patience_trigger,
        ) = generate_residual_mask(data)
        if mask_diffuse_env is not None:
            print("mask diffuse is not none")
            print("batch out", data)
            mask_diffuse_env = mask_diffuse_env.nonzero().squeeze(1)
            print("mask_diffuse_env", len(mask_diffuse_env))
        if mask_diffuse_env is not None and len(mask_diffuse_env) == 0:
            print("mask diffuse is zero length")
            print("batch out", data)
            central_atoms, mask_diffuse_env, patience_trigger = None, None, True
        if mask_diffuse_env is None:
            print("mask diffuse is none")
            print("batch out", data)
            print("ok")
            central_atoms, mask_diffuse_env, patience_trigger = None, None, True
    else:
        print("batch out", data)
        print(f"{p} > 0.95 or {n_atom_min} <= 2")
        central_atoms, mask_diffuse_env, patience_trigger = None, None, True
    print("triggered", patience_trigger)
    data.node_attrs = data.node_attrs / normalization_factor  # Normalize atomic numbers
    edge_index_non_periodic, _ = reconstruct_neighorhood(
        positions=data.positions,
        ptr=data.ptr,
        pbc=torch.zeros_like(data.pbc).bool(),
        cell=data.cell,
        num_graphs=data.num_graphs,
        r_max_nn=r_max_nn,  # TODO: change to r_max_nn
    )
    (positions, node_attrs, eps, mask_env) = add_noise_position_and_attr_local(
        positions=data.positions,
        node_attrs=data.node_attrs,
        edge_index=edge_index_non_periodic,
        num_graphs=data.num_graphs,
        normalization_factor=normalization_factor,
        t=t,
        noise_scheduler=noise_scheduler,
        patience_trigger=patience_trigger,
        batch=data.batch,
        central_atom=central_atoms,
        mask_env=mask_diffuse_env,
    )

    mask_diffuse_env = torch.zeros(
        positions.shape[0], dtype=torch.bool, device=positions.device
    )
    mask_diffuse_env[mask_env] = True
    edge_index, shifts = reconstruct_neighorhood(
        positions=positions,
        ptr=data.ptr,
        pbc=data.pbc,
        cell=data.cell,
        num_graphs=data.num_graphs,
        r_max_nn=max(r_max_nn, r_max),  # TODO: change to r_max_nn
    )

    data.edge_index = edge_index
    data.shifts = shifts
    return (
        positions,
        node_attrs,
        eps,
        mask_env,
        mask_diffuse_env,
        data,
        t,
        t_is_zero,
    )


def index_add(x, y, mask, generation=False):
    if not generation:
        out = torch.index_add(x, 0, mask, y[mask])
    else:
        out = x + y * mask.float().unsqueeze(-1)
    return out

from collections import namedtuple
import copy
from typing import Any, Callable, Dict, List, Optional, Type, Tuple
import ase
from tqdm import tqdm


import torch
from mace import tools

from mace.data import AtomicData
from mace.data.utils import atoms_from_batch
from mace.tools.diffusion_tools import remove_mean, remove_mean_mask
from mace.tools.scatter import scatter_sum
from .utils import (
    SNR_weight,
    get_NN_residual_batch,
    get_central_heavy_atom,
    get_noisy_batch,
    append_noisy_local_environment,
    get_num_heavy_atoms,
    get_sigma_and_alpha_given_s,
    reconstruct_neighorhood,
)


class DiffusionGenerator:
    def __init__(self, model: Callable, node_distribution: Callable):
        self.model = model
        self.node_distribution = node_distribution
        self.noise_scheduler = self.model.noise_scheduler
        self.num_elements = self.model.num_elements
        self.max_steps = self.model.max_steps
        self.z_table = tools.get_atomic_number_table_from_zs([1, 6, 7, 8, 9])
        self.normalization_factor = model.normalization_factor

    def generate(self, n_samples: int) -> List[AtomicData]:
        return self.sample(self.node_distribution.sample(n_samples))

    def sample_p_xh_given_z0(self, data: torch.Tensor) -> Dict[str, Any]:
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(data.num_graphs, 1, device=data.positions.device)
        gamma_0 = self.noise_scheduler(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = torch.exp(0.5 * gamma_0[data.batch])
        out = self.model(data, t=zeros, generation=True, training=False)

        sigma_0 = torch.sqrt(torch.sigmoid(gamma_0[data.batch]))
        alpha_0 = torch.sqrt(1 - torch.sigmoid(gamma_0[data.batch]))
        # Compute mu for p(zs | zt).
        eps_positions = out["predicted_noise_positions"]
        eps_labels = out["predicted_noise_labels"]

        eps_positions = 1.0 / alpha_0 * (data.positions - sigma_0 * eps_positions)
        eps_labels = 1.0 / alpha_0 * (data.node_attrs - sigma_0 * eps_labels)

        positions_out = eps_positions + sigma_x * remove_mean(
            torch.randn_like(eps_positions), data.batch
        )
        labels_out = self.normalization_factor * (
            eps_labels + sigma_x * torch.randn_like(eps_labels)
        )
        labels_out = torch.nn.functional.one_hot(
            torch.argmax(labels_out, dim=-1), self.num_elements
        )
        return positions_out, labels_out

    def sample_p_zs_given_zt(self, s: int, t: int, data: AtomicData) -> Dict[str, Any]:
        gamma_s = self.noise_scheduler(s)
        gamma_t = self.noise_scheduler(t)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = get_sigma_and_alpha_given_s(
            gamma_s=gamma_s, gamma_t=gamma_t, batch=data.batch
        )

        sigma_s = torch.sqrt(torch.sigmoid(gamma_s[data.batch]))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t[data.batch]))

        out = self.model(data, t=t, generation=True, training=False)
        eps_positions = out["predicted_noise_positions"]
        eps_labels = out["predicted_noise_labels"]

        mu_positions = (
            data.positions / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_positions
        )
        mu_labels = (
            data.node_attrs / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_labels
        )  # TODO: unnormalize

        sigma = sigma_t_given_s * sigma_s / sigma_t

        positions_out = mu_positions + sigma * remove_mean(
            torch.randn_like(mu_positions), data.batch
        )
        labels_out = self.normalization_factor * (
            mu_labels + sigma * torch.randn_like(mu_labels)
        )

        return (positions_out, labels_out)

    @torch.no_grad()
    def sample(self, n_nodes):
        """
        Draw samples from the generative model.
        """
        # Initialize
        device = self.noise_scheduler.gamma.device
        data = get_noisy_batch(n_nodes, self.num_elements, device=device)
        generated_config = namedtuple("generated_config", ["positions", "labels"])
        chain = []
        ase_chain = []

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(reversed((range(0, self.max_steps)))):
            s_tensor = torch.tensor(
                [s / self.max_steps] * data.num_graphs, device=device
            ).unsqueeze(-1)
            t_tensor = torch.tensor(
                [(s + 1) / self.max_steps] * data.num_graphs, device=device
            ).unsqueeze(-1)
            positions_out, labels_out = self.sample_p_zs_given_zt(
                s_tensor, t_tensor, data
            )

            data.positions = positions_out
            data.node_attrs = labels_out

            # Write to chain tensor.
            chain.append([generated_config(positions_out, labels_out)])
            ase_chain.append(atoms_from_batch(data, self.z_table))

        # Finally sample p(x, h | z_0).
        positions_out, labels_out = self.sample_p_xh_given_z0(data)

        data.positions = positions_out
        data.node_attrs = labels_out

        return {
            "ase_chain": ase_chain,
            "chain": chain,
            "samples": atoms_from_batch(data, self.z_table),
            "batch": data,
        }


class LocalDiffusionGenerator:
    def __init__(
        self,
        model: Callable,
        node_distribution: Callable,
        skin: int = 0.3,
        termination_model: Optional[Callable] = None,
    ):
        self.model = model
        self.termination_model = termination_model
        self.node_distribution = node_distribution
        self.noise_scheduler = self.model.noise_scheduler
        self.num_elements = self.model.num_elements
        self.max_steps = self.model.max_steps
        self.skin = skin
        self.r_max = self.model.r_max
        self.r_max_nn = self.model.r_max_nn
        # self.z_table = tools.get_atomic_number_table_from_zs([1, 6, 7, 8, 9])
        self.z_table = tools.get_atomic_number_table_from_zs([1, 6])
        self.normalization_factor = model.normalization_factor

    def sample_p_xh_given_z0(
        self, data: AtomicData, central_atoms: torch.Tensor, node_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(data.num_graphs, 1, device=data.positions.device)
        gamma_0 = self.noise_scheduler(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = torch.exp(0.5 * gamma_0[data.batch])
        out = self.model(
            data,
            t=zeros,
            central_atoms=central_atoms,
            generation=True,
            training=False,
            mask_env=node_mask,
        )

        sigma_0 = torch.sqrt(torch.sigmoid(gamma_0[data.batch]))
        alpha_0 = torch.sqrt(1 - torch.sigmoid(gamma_0[data.batch]))
        # Compute mu for p(zs | zt).
        eps_positions = out["predicted_noise_positions"]
        eps_labels = out["predicted_noise_labels"]

        eps_positions = (
            1.0
            / alpha_0
            * (
                data.positions
                - sigma_0 * eps_positions * node_mask.float().unsqueeze(-1)
            )
        )
        eps_labels = (
            1.0
            / alpha_0
            * (data.node_attrs - sigma_0 * eps_labels * node_mask.float().unsqueeze(-1))
        )
        # eps_add = (
        #     sigma_x
        #     * remove_mean_mask(
        #         torch.randn_like(eps_positions) * node_mask.float().unsqueeze(-1),
        #         data.batch,
        #         node_mask,
        #     )
        #     * node_mask.float().unsqueeze(-1)
        # )
        # eps_add = (
        #     sigma_x * torch.randn_like(eps_positions) * node_mask.float().unsqueeze(-1)
        # )
        positions_out = eps_positions
        labels_out = self.normalization_factor * (
            eps_labels
            + sigma_x * torch.randn_like(eps_labels) * node_mask.float().unsqueeze(-1)
        )
        labels_out = torch.nn.functional.one_hot(
            torch.argmax(labels_out, dim=-1), self.num_elements
        ).float()
        return positions_out, labels_out

    def sample_p_zs_given_zt(
        self,
        s: int,
        t: int,
        data: AtomicData,
        central_atoms: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        gamma_s = self.noise_scheduler(s)
        gamma_t = self.noise_scheduler(t)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = get_sigma_and_alpha_given_s(
            gamma_s=gamma_s, gamma_t=gamma_t, batch=data.batch
        )

        sigma_s = torch.sqrt(torch.sigmoid(gamma_s[data.batch]))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t[data.batch]))
        out = self.model(
            data,
            t=t,
            central_atoms=central_atoms,
            generation=True,
            training=False,
            mask_env=node_mask,
        )
        eps_positions = out["predicted_noise_positions"]
        eps_labels = out["predicted_noise_labels"]
        # print("force no out", data.positions / alpha_t_given_s - data.positions)
        mu_positions_add = data.positions / alpha_t_given_s - (
            3.5 * sigma2_t_given_s / alpha_t_given_s / sigma_t
        ) * eps_positions * node_mask.float().unsqueeze(-1)
        mu_positions = data.positions * (~node_mask.bool()).float().unsqueeze(
            -1
        ) + mu_positions_add * node_mask.float().unsqueeze(-1)
        mu_labels_add = data.node_attrs / alpha_t_given_s - (
            sigma2_t_given_s / alpha_t_given_s / sigma_t
        ) * eps_labels * node_mask.float().unsqueeze(-1)
        mu_labels = data.node_attrs * (~node_mask.bool()).float().unsqueeze(
            -1
        ) + mu_labels_add * node_mask.float().unsqueeze(-1)

        sigma = sigma_t_given_s * sigma_s / sigma_t

        eps_add = (
            0.4
            * sigma
            * torch.randn_like(mu_positions)
            * node_mask.float().unsqueeze(-1)
        )
        positions_out = mu_positions + eps_add
        eps_add_labels = (
            sigma * torch.randn_like(mu_labels) * node_mask.float().unsqueeze(-1)
        )
        labels_out = self.normalization_factor * (mu_labels + eps_add_labels)
        return (positions_out, labels_out)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        num_steps: int,
        pbc: Optional[Tuple[bool]] = (False, False, False),
        scaffold_data: Optional[AtomicData] = None,
        central_atoms: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Draw from local diffusion model, for step in range num_steps
        """
        # Initialize first local environment
        generated_config = namedtuple("generated_config", ["positions", "labels"])
        chain = []
        ase_chain = []
        device = self.noise_scheduler.gamma.device
        n_nodes_total = []
        data = scaffold_data
        restarted = False
        failed_central_atoms = []
        step = 0

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        while step < num_steps:
            print("step", step)
            n_nodes = self.node_distribution.sample(n_samples).to(device)
            # n_nodes = torch.ones_like(n_nodes) * 4
            n_nodes_total.append(n_nodes)
            chain.append([])
            ase_chain.append([])
            if data is None:
                data_copy = None
                data = get_noisy_batch(
                    n_nodes=n_nodes,
                    num_elements=self.num_elements,
                    cutoff=self.r_max_nn,
                    pbc=pbc,
                    device=device,
                )
                central_atoms = get_central_heavy_atom(
                    node_attrs=data.node_attrs,
                    batch=data.batch,
                    num_graphs=data.num_graphs,
                )
                node_mask = torch.ones_like(data.node_attrs[:, 0]).bool()
                node_mask[central_atoms] = False
                data.positions = (
                    data.positions - data.positions[central_atoms][data.batch]
                )
                central_distance = torch.linalg.norm(
                    data.positions - data.positions[central_atoms][data.batch],
                    dim=-1,
                )
                rescaling = torch.max(
                    torch.ones_like(central_distance), central_distance / self.r_max_nn
                )
                data.positions = data.positions / rescaling[:, None]
                data.node_attrs[central_atoms, :] = torch.nn.functional.one_hot(
                    torch.tensor([1], device=device),
                    self.num_elements,
                ).float()
                restarted = False
            else:
                if central_atoms is None or step != 0 or restarted:
                    central_atoms = get_central_heavy_atom(
                        node_attrs=data.node_attrs,
                        batch=data.batch,
                        num_graphs=data.num_graphs,
                    ).to(device)
                    print(failed_central_atoms)
                    if restarted and central_atoms in failed_central_atoms:
                        num_heavy_elements = get_num_heavy_atoms(
                            node_attrs=data.node_attrs,
                            batch=data.batch,
                            num_graphs=data.num_graphs,
                        )
                        if len(failed_central_atoms) >= num_heavy_elements.int().item():
                            print("All central atoms have been tried, breaking")
                            break
                        # If we have restarted and the central atoms are the same, pick a new one until different
                        patience = 0
                        while central_atoms in failed_central_atoms:
                            central_atoms = get_central_heavy_atom(
                                node_attrs=data.node_attrs,
                                batch=data.batch,
                                num_graphs=data.num_graphs,
                            ).to(device)
                            patience += 1
                            if patience > 100:
                                print("Patience exceeded, breaking")
                                break
                    restarted = False

                mask_env = (
                    scatter_sum(
                        src=torch.isin(data.edge_index[0], central_atoms).float(),
                        index=data.edge_index[1],
                        dim=0,
                        dim_size=data.num_nodes,
                    )
                    > 0
                )
                print("n_nodes", n_nodes)
                print("central_atoms", central_atoms)
                if self.termination_model is not None:
                    print("termination_model")
                    predicted_numbers = (
                        self.termination_model(data, generation=True)["output"]
                        .softmax(dim=-1)
                        .argmax(dim=-1)
                    )
                    print("predicted_numbers", predicted_numbers)
                    central_atoms = torch.argmax(predicted_numbers, keepdim=True)
                    print("central_atoms", central_atoms)
                    n_nodes = torch.max(predicted_numbers[central_atoms], n_nodes)
                    print("n_nodes", n_nodes)
                data_copy = copy.deepcopy(data)
                data, node_mask = append_noisy_local_environment(
                    batch=data.to(device),
                    n_nodes=n_nodes,
                    cutoff=self.r_max_nn,
                    num_elements=self.num_elements,
                    central_atoms=central_atoms,
                    device=device,
                )
                node_mask = node_mask.bool()
                print("data_copy", data_copy)
                print("data", data)
            # print("node_mask", node_mask)
            # print("positions_input", data.positions)
            # print("node_attrs_input", data.node_attrs)
            print(data)
            print(self.z_table)
            ase_chain[step].append(atoms_from_batch(data, self.z_table))
            positions_init = data.positions
            for s in tqdm(reversed((range(0, self.max_steps)))):
                s_tensor = torch.tensor(
                    [s / self.max_steps] * data.num_graphs, device=device
                ).unsqueeze(-1)
                t_tensor = torch.tensor(
                    [(s + 1) / self.max_steps] * data.num_graphs, device=device
                ).unsqueeze(-1)
                positions_out, labels_out = self.sample_p_zs_given_zt(
                    s_tensor, t_tensor, data, central_atoms, node_mask
                )

                data.positions = positions_out
                shift = (data.positions - positions_init).abs()
                if data.positions.isnan().any() or (shift > 10).any():
                    if data.positions.isnan().any():
                        print("Nan in positions")
                    elif (shift > 10).any():
                        print("Exploded!")
                    data = data_copy
                    restarted = True
                    # if the generation fails, add the central atoms to the list of failed atoms
                    print(failed_central_atoms)
                    print(step)
                    if central_atoms not in failed_central_atoms and step != 0:
                        failed_central_atoms.append(central_atoms)
                    break
                edge_index, shifts = reconstruct_neighorhood(
                    positions=data.positions,
                    ptr=data.ptr,
                    pbc=data.pbc,
                    cell=data.cell,
                    num_graphs=data.num_graphs,
                    r_max_nn=self.r_max_nn,  # TODO: change to r_max_nn
                )
                data.edge_index = edge_index
                data.shifts = shifts
                data.node_attrs = labels_out
                # Write to chain tensor.
                chain[step].append([generated_config(positions_out, labels_out)])
                ase_chain[step].append(atoms_from_batch(data, self.z_table))

            # # Finally sample p(x, h | z_0).

            if not restarted:
                positions_out, labels_out = self.sample_p_xh_given_z0(
                    data, central_atoms, node_mask
                )
                data.positions = positions_out
                data.node_attrs = labels_out
                print("labels_out", labels_out)
                step += 1
                if step != num_steps:
                    data = get_NN_residual_batch(data, central_atoms, node_mask).to(
                        device
                    )
                    print("data", data)

        return {
            "chain": chain,
            "ase_chain": ase_chain,
            "samples": atoms_from_batch(data, self.z_table),
        }

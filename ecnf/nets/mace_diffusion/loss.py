###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the ASL License (see ASL.md)
###########################################################################################

from typing import Optional
import torch

from mace.tools import TensorDict
from mace.tools.scatter import scatter_sum, scatter_mean
from mace.tools.torch_geometric import Batch


def mean_squared_error_eps_positions(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # eps_positions: [n_atoms, 3]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    return scatter_mean(
        torch.mean(
            configs_weight
            * torch.square(pred["noise_positions"] - pred["predicted_noise_positions"]),
            dim=-1,
        ),
        ref.batch,
        dim=0,
    )


def mean_squared_error_eps_labels(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # eps_labels: [n_atoms, 3]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    return scatter_mean(
        torch.mean(
            configs_weight
            * torch.square(pred["noise_labels"] - pred["predicted_noise_labels"]),
            dim=-1,
        ),
        ref.batch,
        dim=0,
    )


def KL_distance_gaussian(
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_p: torch.Tensor,
    std_p: torch.Tensor,
    d: Optional[int] = None,
) -> torch.Tensor:
    if d is None:
        return torch.sum(
            (torch.square(std_p) + torch.square(mean_q - mean_p)) / torch.square(std_p)
            - 1
            - 2 * torch.log(std_q / std_p),
            dim=-1,
        )  # TODO: Use scatter sum
    else:
        return torch.sum(
            (
                d * torch.log(std_p / std_q)
                + 0.5 * d * torch.square(std_q) / torch.square(std_p)
                + 0.5 * d * torch.square(mean_q - mean_p) / torch.square(std_p)
                - 0.5 * d
            ),
            dim=-1,
        )


def kl_prior_loss(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # eps_labels: [n_atoms, 3]
    gamma_T = pred["noise_scheduler"](torch.ones(ref.num_graphs, 1))
    alpha_T = torch.sqrt(torch.sigmoid(-gamma_T[ref.batch]))
    sigma_T = torch.sqrt(torch.sigmoid(gamma_T[ref.batch]))
    mean_positions, mean_labels = (
        alpha_T * pred["predicted_noise_positions"],
        alpha_T * pred["predicted_noise_labels"],
    )
    KL_distance_labels = 0.5 * KL_distance_gaussian(
        mean_q=mean_labels,
        std_q=sigma_T,
        mean_p=torch.zeros_like(mean_labels),
        std_p=torch.ones_like(sigma_T),
    )
    num_nodes = (ref.ptr[1:] - ref.ptr[:-1])[ref.batch].unsqueeze(-1)
    KL_distance_positions = KL_distance_gaussian(
        mean_q=mean_positions,
        std_q=sigma_T,
        mean_p=torch.zeros_like(mean_positions),
        std_p=torch.ones_like(sigma_T),
        d=(num_nodes - 1) * 3,
    )
    return scatter_mean(KL_distance_labels + KL_distance_positions, ref.batch, dim=0)


def log_constant_loss(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # Compute p(x | z0)
    num_nodes = ref.ptr[1:] - ref.ptr[:-1]
    gamma_0 = pred["noise_scheduler"](torch.zeros(ref.num_graphs, 1)).squeeze(-1)
    return ((num_nodes - 1) * 3) * (
        -0.5 * gamma_0 - 0.5 * torch.log(2 * torch.tensor(2 * torch.pi))
    )


def log_without_constants_loss(ref: Batch, pred: TensorDict) -> torch.Tensor:
    gamma_0 = pred["noise_scheduler"](torch.zeros(ref.num_graphs, 1))
    sigma_0 = torch.sqrt(torch.sigmoid(gamma_0[ref.batch]))
    error_positions = -0.5 * mean_squared_error_eps_positions(ref, pred)
    centered_labels_noise = (
        pred["predicted_noise_labels"] - 1
    )  # Centered around one for true label
    integral_label = torch.log(
        0.5
        * torch.erf(
            (centered_labels_noise + 0.5) / (sigma_0 * torch.sqrt(torch.tensor(2.0)))
        )
        - 0.5
        * torch.erf(
            (centered_labels_noise - 0.5) / (sigma_0 * torch.sqrt(torch.tensor(2.0)))
        )
        + 1e-10
    ).sum(dim=-1, keepdim=True)
    # Normalize integral
    log_Z = torch.logsumexp(integral_label, dim=-1, keepdim=True)
    log_probabilities_label = scatter_sum(
        torch.sum((integral_label - log_Z) * ref.node_attrs, dim=-1), ref.batch, dim=0
    )
    return error_positions + log_probabilities_label


# class PositionsLabelsLoss(torch.nn.Module):
#     def __init__(self, weight_positions: float = 1.0, weight_labels: float = 1.0):
#         super().__init__()
#         self.weight_positions = weight_positions
#         self.weight_labels = weight_labels

#     def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
#         loss_error = (
#             0.5
#             * (
#                 self.weight_positions * mean_squared_error_eps_positions(ref, pred)
#                 + self.weight_labels * mean_squared_error_eps_labels(ref, pred)
#             )
#             * (1 - pred["t_is_zero"])
#             + log_without_constants_loss(ref, pred) * pred["t_is_zero"]
#         )
#         return torch.mean(
#             loss_error + kl_prior_loss(ref, pred) + log_constant_loss(ref, pred)
#         )


class PositionsLabelsLoss(torch.nn.Module):
    def __init__(self, weight_positions: float = 1.0, weight_labels: float = 1.0):
        super().__init__()
        self.weight_positions = weight_positions
        self.weight_labels = weight_labels

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return torch.mean(pred["loss"])


from abc import ABC
from typing import Callable, Dict, Optional, Tuple, Union
import math

from e3nn import o3

import e3nn_jax as e3nn
import flax
import jax.numpy as jnp
import chex

from mace.tools.scatter import scatter_sum

from .irreps_tools_jax import (reshape_irreps, tp_out_irreps_with_instructions)
from .symmetric_contraction import SymmetricContraction


class DiffusionInteractionBlock(ABC, flax.linen.Module):
    node_attrs_irreps: e3nn.Irreps
    node_feats_irreps: e3nn.Irreps
    edge_attrs_irreps: e3nn.Irreps
    edge_feats_irreps: e3nn.Irreps
    target_irreps: e3nn.Irreps
    hidden_irreps: e3nn.Irreps
    avg_num_neighbors: float
    r_max: Optional[float] = None

    def setup(self):
        self.setup_jax()

    def setup_jax(self):    
        # First linear
        irreps_scalar = e3nn.Irreps(
            [(self.node_feats_irreps.count(e3nn.Irrep(0, 1)), (0, 1))]
        )

        self.linear_scalar = e3nn.flax.Linear(irreps_in=self.node_feats_irreps,
                                              irreps_out=irreps_scalar,
                                              path_normalization="element")

        self.linear_up = e3nn.flax.Linear(irreps_in=self.node_feats_irreps,
                                          irreps_out=self.node_feats_irreps,
                                          path_normalization="element")

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )

        irreps_in1 = self.node_feats_irreps
        irreps_in2 = self.edge_attrs_irreps
        self.conv_tp = e3nn.legacy.FunctionalTensorProduct(irreps_in1=irreps_in1,
                                                           irreps_in2=irreps_in2,
                                                           irreps_out=irreps_mid,
                                                           instructions=instructions,
                                                           irrep_normalization="component",
                                                           path_normalization="element")

        # Convolution weights
        input_dim = self.hidden_irreps.count(e3nn.Irrep(0, 1))
        weight_numel = sum(math.prod((irreps_in1[ins[0]].mul, irreps_in2[ins[1]].mul)) 
                           for ins in instructions if ins[4]
                           )
        layer = flax.linen.Dense(features=weight_numel, 
                                 use_bias=False, 
                                 kernel_init=flax.linen.initializers.variance_scaling(scale=(0.001)**2, mode="fan_avg", distribution="uniform")
                                 )

        # Selector TensorProduct
        self.conv_tp_weights = flax.linen.Sequential(
            [flax.linen.Dense(features=input_dim),
             flax.linen.silu,
             flax.linen.Dense(features=input_dim),
             flax.linen.silu,
             layer]
            )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = e3nn.flax.Linear(irreps_in=irreps_mid,
                                       irreps_out=self.irreps_out,
                                       path_normalization="element")

        self.reshape = reshape_irreps(self.irreps_out)



    def __call__(
        self,
        node_feats: chex.Array,
        edge_attrs: chex.Array,
        edge_feats: chex.Array,
        lengths: chex.Array,
        edge_index: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]
        node_scalars = self.linear_scalar(node_feats)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(
            # torch.cat(
            #     [node_scalars[sender], node_scalars[receiver], edge_feats, lengths],
            #     dim=-1,
            # )
            jnp.concatenate(
                [node_scalars[sender].array, node_scalars[receiver].array, edge_feats, lengths],
                axis=-1,
            )

        )
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


class EquivariantProductBasisBlock(flax.linen.Module):
    node_feats_irreps: e3nn.Irreps
    target_irreps: e3nn.Irreps
    correlation: Union[int, Dict[str, int]]
    element_dependent: bool = True
    use_sc: bool = True
    num_species: Optional[int] = None

    def setup(self):
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.target_irreps,
            correlation=self.correlation,
            element_dependent=self.element_dependent,
            num_species=self.num_species,
        )
        # Update linear
        self.linear = o3.Linear(
            self.target_irreps,
            self.target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def __call__(
        self, 
        node_feats: chex.Array, 
        sc: chex.Array, 
        node_attrs: chex.Array
    ) -> chex.Array:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc:
            return self.linear(node_feats) + sc

        return self.linear(node_feats)


class NonLinearReadoutBlock(flax.linen.Module):
    irreps_in: e3nn.Irreps
    MLP_irreps: e3nn.Irreps
    gate: Callable
    num_species: int
    positions_only: bool = False
    charges: bool = False

    def setup(self):
        if self.positions_only:
            self.irreps_out = e3nn.Irreps("1x1o")
        if self.charges:
            self.irreps_out = e3nn.Irreps(str(self.num_species + 1) + "x0e + 1x1o")
        else:
            self.irreps_out = e3nn.Irreps(str(self.num_species) + "x0e + 1x1o")
        irreps_scalars = e3nn.Irreps(
            [(mul, ir) for mul, ir in self.MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = e3nn.Irreps(
            [(mul, ir) for mul, ir in self.MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = e3nn.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[self.gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[self.gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = o3.Linear(
            irreps_in=self.irreps_in, irreps_out=self.irreps_nonlin, biases=True
        )
        self.linear_2 = o3.Linear(
            irreps_in=self.MLP_irreps,
            irreps_out=self.irreps_out,
            biases=True,
        )

    def forward(self, x: chex.Array) -> chex.Array:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]

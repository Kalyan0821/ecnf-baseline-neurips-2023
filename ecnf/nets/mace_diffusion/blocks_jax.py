from typing import Callable, Dict, Optional, Tuple, Union
import e3nn_jax as e3nn
import flax
import jax.numpy as jnp
import chex
from .irreps_tools_jax import reshape_irreps, tp_out_irreps_with_instructions

import haiku as hk
import haiku.experimental.flax as hkflax
from ecnf.nets.mace_modules.symmetric_contraction import SymmetricContraction
from ecnf.nets.mace_modules.symmetric_contraction_flax import SymmetricContractionFlax


class DiffusionInteractionBlock(flax.linen.Module):
    node_attrs_irreps: e3nn.Irreps
    node_feats_irreps: e3nn.Irreps
    edge_attrs_irreps: e3nn.Irreps
    edge_feats_irreps: e3nn.Irreps
    target_irreps: e3nn.Irreps
    hidden_irreps: e3nn.Irreps
    avg_num_neighbors: float
    variance_scaling_init: float


    def setup(self):
        node_attrs_irreps = e3nn.Irreps(self.node_attrs_irreps)
        node_feats_irreps = e3nn.Irreps(self.node_feats_irreps)
        edge_attrs_irreps = e3nn.Irreps(self.edge_attrs_irreps)
        edge_feats_irreps = e3nn.Irreps(self.edge_feats_irreps)
        target_irreps = e3nn.Irreps(self.target_irreps)
        hidden_irreps = e3nn.Irreps(self.hidden_irreps)        

        # First linear
        irreps_scalar = e3nn.Irreps(
            [(node_feats_irreps.count(e3nn.Irrep(0, 1)), (0, 1))]
        )

        self.linear_scalar = e3nn.flax.Linear(irreps_in=node_feats_irreps,
                                              irreps_out=irreps_scalar,
                                              path_normalization="element")

        self.linear_up = e3nn.flax.Linear(irreps_in=node_feats_irreps,
                                          irreps_out=node_feats_irreps,
                                          path_normalization="element")

        input_dim = hidden_irreps.count(e3nn.Irrep(0, 1))

        # TensorProduct convolution weights
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            edge_attrs_irreps,
            target_irreps)
        irreps_mid = e3nn.Irreps(irreps_mid)
        self.irreps_mid = irreps_mid

        layer = flax.linen.Dense(features=irreps_mid.num_irreps, 
                                 use_bias=False, 
                                 kernel_init=flax.linen.initializers.variance_scaling(scale=self.variance_scaling_init, 
                                                                                      mode="fan_avg", 
                                                                                      distribution="uniform")
                                 )
        self.conv_tp_weights = flax.linen.Sequential(
                [flax.linen.Dense(features=input_dim),
                 flax.linen.silu,
                 flax.linen.Dense(features=input_dim),
                 flax.linen.silu,
                 layer]
            )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = target_irreps
        self.linear = e3nn.flax.Linear(irreps_in=irreps_mid,
                                       irreps_out=self.irreps_out,
                                       path_normalization="element")

    def __call__(
        self,
        node_feats: chex.Array,
        edge_attrs: chex.Array,
        edge_feats: chex.Array,
        lengths: chex.Array,
        edge_index: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        
        node_scalars = self.linear_scalar(node_feats)
        node_feats = self.linear_up(node_feats)

        sender, receiver = edge_index
        tp_weights = self.conv_tp_weights(
            jnp.concatenate(
                [node_scalars[sender].array, node_scalars[receiver].array, edge_feats, lengths],
                axis=-1)
        )

        mji = e3nn.tensor_product(input1=node_feats[sender],
                                  input2=edge_attrs,
                                  filter_ir_out=e3nn.Irreps(self.irreps_mid)
                                  )
        mji *= tp_weights
        
        num_nodes = node_feats.shape[0]
        message = e3nn.scatter_sum(data=mji, dst=receiver, output_size=num_nodes)  # (n_nodes, irreps)

        message = self.linear(message) / self.avg_num_neighbors

        return (
            reshape_irreps(message, e3nn.Irreps(self.irreps_out)),
            None,
        )  # (n_nodes, channels, (lmax + 1)**2)


class EquivariantProductBasisBlock(flax.linen.Module):
    node_feats_irreps: e3nn.Irreps
    target_irreps: e3nn.Irreps
    correlation: Union[int, Dict[str, int]]
    num_species: Optional[int] = None

    element_dependent: bool = True
    use_sc: bool = True
    use_library_contraction: bool = False

    def setup(self):
        node_feats_irreps = e3nn.Irreps(self.node_feats_irreps)
        target_irreps = e3nn.Irreps(self.target_irreps)

        if self.use_library_contraction:  
            print("Using existing haiku contraction...")
            symmetric_contraction = hk.transform(
                lambda x, y: SymmetricContraction(correlation=self.correlation,
                                                  keep_irrep_out=target_irreps,
                                                  num_species=self.num_species,
                                                )(x, y)
                                            )
            symmetric_contraction = hkflax.Module(symmetric_contraction)  # convert to flax module
        else:  
            print("Using my flax contraction...")
            symmetric_contraction = SymmetricContractionFlax(
                correlation=self.correlation,
                keep_irrep_out=target_irreps,
                num_species=self.num_species,
            )

        self.symmetric_contraction = symmetric_contraction

        # Update linear
        self.linear = e3nn.flax.Linear(
            irreps_in=target_irreps,
            irreps_out=target_irreps,
            path_normalization="element",
        )

    def __call__(
        self, 
        node_feats: chex.Array, 
        node_attrs: chex.Array,
        sc: chex.Array, 
    ) -> chex.Array:
        
        node_feats = self.symmetric_contraction(node_feats, node_attrs)
        node_feats = node_feats.axis_to_mul()
        # linear
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
        irreps_in = e3nn.Irreps(self.irreps_in)
        MLP_irreps = e3nn.Irreps(self.MLP_irreps)

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
        
        self.equivariant_nonlin = lambda x: e3nn.gate(input=x,
                                                      even_act=self.gate,
                                                      odd_act=self.gate,
                                                      even_gate_act=self.gate,
                                                      odd_gate_act=self.gate,
                                                      normalize_act=True,
                                                )
                                                      
        self.irreps_nonlin = sum(tuple(e3nn.Irreps(irreps).simplify() for irreps in [irreps_scalars, irreps_gates, irreps_gated]), e3nn.Irreps([])).sort()[0].simplify()
        self.irreps_nonlin = e3nn.Irreps(self.irreps_nonlin)

        self.linear_1 = e3nn.flax.Linear(irreps_in=irreps_in,
                                         irreps_out=self.irreps_nonlin,
                                         biases=True,
                                         path_normalization="element")

        self.linear_2 = e3nn.flax.Linear(irreps_in=MLP_irreps,
                                         irreps_out=self.irreps_out,
                                         biases=True,
                                         path_normalization="element")

    def __call__(self, x: chex.Array) -> chex.Array:  # [n_nodes, irreps]
        x = self.linear_1(x)
        x = self.equivariant_nonlin(x)
        x = self.linear_2(x)
        return x  # [n_nodes, 1]

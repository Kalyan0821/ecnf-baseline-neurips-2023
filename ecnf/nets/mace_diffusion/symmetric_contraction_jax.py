from typing import Dict, Optional, Union

import jax.numpy as jnp
import flax
import chex
import e3nn_jax as e3nn
from opt_einsum import contract
from mace.tools.cg import U_matrix_real

class SymmetricContraction(flax.linen.Module):
    irreps_in: e3nn.Irreps
    irreps_out: e3nn.Irreps
    correlation: Union[int, Dict[str, int]]
    num_species: Optional[int] = None

    element_dependent: Optional[bool] = None
    irrep_normalization: str = "component"
    path_normalization: str = "element"
    internal_weights: Optional[bool] = None
    shared_weights: Optional[chex.Array] = None

    def setup(self):
        assert self.irrep_normalization in ["component", "norm", "none"]
        assert self.path_normalization in ["element", "path", "none"]

        irreps_in = e3nn.Irreps(self.irreps_in)
        irreps_out = e3nn.Irreps(self.irreps_out)

        if not isinstance(self.correlation, tuple):
            corr = self.correlation
            correlation = {}
            for irrep_out in irreps_out:
                correlation[irrep_out] = corr

        shared_weights, internal_weights, element_dependent = self.shared_weights, self.internal_weights, self.element_dependent
        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        if element_dependent is None:
            element_dependent = True

        contractions = dict()
        for irrep_out in irreps_out:
            contractions[str(irrep_out)] = Contraction(
                irreps_in=irreps_in,
                irrep_out=e3nn.Irreps(str(irrep_out.ir)),
                correlation=correlation[irrep_out],
                internal_weights=internal_weights,
                element_dependent=element_dependent,
                num_species=self.num_species,
                weights=shared_weights,
            )
        self.contractions = contractions

    def __call__(self, x: chex.Array, y: chex.Array):
        outs = []
        for irrep in self.irreps_out:
            outs.append(self.contractions[str(irrep)](x, y))

        return jnp.concatenate(outs, axis=-1)


class Contraction(flax.linen.Module):
    irreps_in: e3nn.Irreps
    irrep_out: e3nn.Irreps
    correlation: int
    internal_weights: bool = True
    element_dependent: bool = True
    num_species: Optional[int] = None
    weights: Optional[chex.Array] = None

    def setup(self):
        self.num_features = self.irreps_in.count((0, 1))

        U_matrices = dict()
        for nu in range(1, self.correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=repr(e3nn.Irreps([irrep.ir for irrep in self.irreps_in])),
                irreps_out=repr(e3nn.Irreps(self.irrep_out)),
                correlation=nu,
                # dtype=dtype,
            )[-1]
            U_matrices[nu] = jnp.array(U_matrix)

        self.U_matrices = U_matrices

        if self.element_dependent:
            # Tensor contraction equations
            self.equation_main = "...ik,ekc,bci,be -> bc..."
            self.equation_weighting = "...k,ekc,be->bc..."
            self.equation_contract = "bc...i,bci->bc..."
            if self.internal_weights:
                # Create weight for product basis
                internal_model_weights = dict()
                for i in range(1, self.correlation + 1):
                    num_params = self.U_tensors(i).shape[-1]
                    internal_model_weights[str(i)] = self.param(f"weights_{i}", 
                                           flax.linen.initializers.normal(stddev=1/jnp.sqrt(num_params)), 
                                           (self.num_species, num_params, self.num_features)
                                        )
                self.internal_model_weights = internal_model_weights
            
            else:
                self.register_buffer("weights", self.weights)

        else:
            # Tensor contraction equations
            self.equation_main = "...ik,kc,bci -> bc..."
            self.equation_weighting = "...k,kc->c..."
            self.equation_contract = "bc...i,bci->bc..."
            if self.internal_weights:
                # Create weight for product basis
                internal_model_weights = dict()
                for i in range(1, self.correlation + 1):
                    num_params = self.U_tensors(i).shape[-1]
                    internal_model_weights[str(i)] = self.param(f"weights_{i}", 
                                                                flax.linen.initializers.normal(stddev=1/jnp.sqrt(num_params)), 
                                                                (num_params, self.num_features)
                                        )
                self.internal_model_weights = internal_model_weights

            else:
                self.register_buffer("weights", self.weights)

    def U_tensors(self, nu):
        return self.U_matrices[nu]

    def __call__(self, x: chex.Array, y: Optional[chex.Array]):
        if self.element_dependent:
            out = contract(
                self.equation_main,
                self.U_tensors(self.correlation),
                self.internal_model_weights[str(self.correlation)],
                x,
                y,
            )  # TODO: use optimize library and cuTENSOR  # pylint: disable=fixme
            for corr in range(self.correlation - 1, 0, -1):
                c_tensor = contract(
                    self.equation_weighting,
                    self.U_tensors(corr),
                    self.internal_model_weights[str(corr)],
                    y,
                )
                c_tensor = c_tensor + out
                out = contract(self.equation_contract, c_tensor, x)

        else:
            out = contract(
                self.equation_main,
                self.U_tensors(self.correlation),
                self.internal_model_weights[str(self.correlation)],
                x,
            )  # TODO: use optimize library and cuTENSOR  # pylint: disable=fixme
            for corr in range(self.correlation - 1, 0, -1):
                c_tensor = contract(
                    self.equation_weighting,
                    self.U_tensors(corr),
                    self.internal_model_weights[str(corr)],
                )
                c_tensor = c_tensor + out
                out = contract(self.equation_contract, c_tensor, x)
        resize_shape = torch.prod(jnp.array(out.shape[1:]))
        return out.view(out.shape[0], resize_shape)



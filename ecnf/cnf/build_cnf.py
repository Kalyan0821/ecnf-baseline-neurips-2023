"""Build's CNF for application to Cartesian coordinates of molecules."""
from typing import Sequence

from functools import partial

import jax.numpy as jnp
import distrax
import jax

from ecnf.cnf.core import FlowMatchingCNF, optimal_transport_conditional_vf
from ecnf.cnf.zero_com_base import FlatZeroCoMGaussian
from ecnf.cnf.flat_models import FlatEGNN, FlatMACEDiffusion


def build_cnf(
        n_nodes: int,
        dim: int,
        sigma_min: float,
        base_scale: float,
        model_name: str,
        n_features: int,
        time_embedding_dim: int,
        # egnn specific
        n_invariant_feat_hidden: int,
        n_blocks: int,
        mlp_units: Sequence[int],
        # mace specific
        readout_mlp_irreps: str,
        hidden_irreps: str,
        num_interactions: int,
        graph_type: str,          
        avg_num_neighbors: str,  
        max_ell: int,
        variance_scaling_init: float,
        correlation: int,
        zero_com: bool,
        scale_output: bool,
):

    scale_bijector = distrax.ScalarAffine(
        shift=jnp.zeros(dim*n_nodes),
        scale=jnp.ones(dim*n_nodes)*base_scale)

    scale_bijector_zero_com = distrax.Lambda(
        forward=scale_bijector.forward,
        inverse=scale_bijector.inverse,
        forward_log_det_jacobian=lambda x: jnp.sum(scale_bijector.forward_log_det_jacobian(x), axis=-1) * (n_nodes - 1) / n_nodes,
        inverse_log_det_jacobian=lambda y: jnp.sum(scale_bijector.inverse_log_det_jacobian(y), axis=-1) * (n_nodes - 1) / n_nodes,
        event_ndims_in=1,
        event_ndims_out=1
    )

    base = distrax.Transformed(
        distribution=FlatZeroCoMGaussian(dim=dim, n_nodes=n_nodes),
        bijector=scale_bijector_zero_com)

    get_cond_vector_field = partial(optimal_transport_conditional_vf, sigma_min=sigma_min)

    assert n_features == 1  # we are only generating atom positions, not atom types too
    if model_name == "egnn":
        net = FlatEGNN(n_nodes=n_nodes,
                       dim=dim,
                       n_features=n_features,
                       n_invariant_feat_hidden=n_invariant_feat_hidden,
                       time_embedding_dim=time_embedding_dim,
                       n_blocks=n_blocks,
                       mlp_units=mlp_units
        )
    elif model_name == "mace":
        net = FlatMACEDiffusion(n_nodes=n_nodes,
                                dim=dim,
                                n_features=n_features,
                                time_embedding_dim=time_embedding_dim,
                                hidden_irreps=hidden_irreps,
                                readout_mlp_irreps=readout_mlp_irreps,
                                num_interactions=num_interactions,
                                num_species=int(n_features),
                                graph_type=graph_type,
                                avg_num_neighbors=avg_num_neighbors,
                                max_ell=max_ell,
                                variance_scaling_init=variance_scaling_init,
                                correlation=correlation,
                                zero_com=zero_com,
                                scale_output=scale_output,
        )
    else:
        raise NotImplementedError
    
    print("--------------------------------------------")
    print("Num. params:")
    params = net.init(jax.random.PRNGKey(0), 
                      jnp.zeros((1, n_nodes*dim)), 
                      jnp.zeros((1,)), 
                      jnp.zeros((1, n_nodes*n_features), dtype=int)
            )
    print(sum(x.size for x in jax.tree_leaves(params)))
    print("--------------------------------------------")
    # exit()

    cnf = FlowMatchingCNF(init=net.init, apply=net.apply, get_x_t_and_conditional_u_t=get_cond_vector_field,
                          sample_base=base._sample_n, sample_and_log_prob_base=base.sample_and_log_prob,
                          log_prob_base=base.log_prob)

    return cnf

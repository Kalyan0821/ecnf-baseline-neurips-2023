"""Build's CNF for application to Cartesian coordinates of molecules."""
from typing import Sequence

from functools import partial

import jax.numpy as jnp
import distrax
import jax

from ecnf.cnf.core import FlowMatchingCNF, optimal_transport_conditional_vf
from ecnf.cnf.zero_com_base import FlatZeroCoMGaussian
from ecnf.cnf.flat_models import FlatEGNN, FlatMACE, FlatMACEDiffusion


def build_cnf(
        n_frames: int,
        dim: int,
        sigma_min: float,
        base_scale: float,
        n_blocks_egnn: int,
        mlp_units: Sequence[int],
        n_invariant_feat_hidden: int,
        time_embedding_dim: int,
        n_features: int,
        model_name: str,
        # mace specific
        output_irreps: str,
        readout_mlp_irreps: str,
        hidden_irreps: str,
        r_max: float,  
        num_interactions: int,
        epsilon: float,
        graph_type: str,          
        avg_num_neighbors: str,  
        output_mode: str
):

    scale_bijector = distrax.ScalarAffine(
        shift=jnp.zeros(dim*n_frames),
        scale=jnp.ones(dim*n_frames)*base_scale)

    scale_bijector_zero_com = distrax.Lambda(
        forward=scale_bijector.forward,
        inverse=scale_bijector.inverse,
        forward_log_det_jacobian=lambda x: jnp.sum(scale_bijector.forward_log_det_jacobian(x), axis=-1) * (n_frames - 1) / n_frames,
        inverse_log_det_jacobian=lambda y: jnp.sum(scale_bijector.inverse_log_det_jacobian(y), axis=-1) * (n_frames - 1) / n_frames,
        event_ndims_in=1,
        event_ndims_out=1
    )

    base = distrax.Transformed(
        distribution=FlatZeroCoMGaussian(dim=dim, n_nodes=n_frames),
        bijector=scale_bijector_zero_com)

    get_cond_vector_field = partial(optimal_transport_conditional_vf, sigma_min=sigma_min)

    assert n_features == 1  # we are only generating atom positions, not atom types too
    if model_name == "egnn":
        net = FlatEGNN(n_nodes=n_frames,
                       dim=dim,
                       n_features=n_features,
                       n_invariant_feat_hidden=n_invariant_feat_hidden,
                       time_embedding_dim=time_embedding_dim,
                       n_blocks_egnn=n_blocks_egnn,
                       mlp_units=mlp_units
        )
    elif model_name == "mace":
        net = FlatMACE(n_nodes=n_frames,
                       dim=dim,
                       n_features=n_features,
                       time_embedding_dim=time_embedding_dim,
                       output_irreps=output_irreps,
                       readout_mlp_irreps=readout_mlp_irreps,
                       hidden_irreps=hidden_irreps,
                       r_max=r_max,
                       num_interactions=num_interactions,
                       epsilon=epsilon,
                       train_graphs=None,        # TODO: get this
                       num_species=int(n_features),
                       graph_type=graph_type,
                       avg_num_neighbors=avg_num_neighbors,
                       output_mode=output_mode
        )
    elif model_name == "mace_diffusion":
        net = FlatMACEDiffusion(n_nodes=n_frames,
                                dim=dim,
                                n_features=n_features,
                                time_embedding_dim=time_embedding_dim,
                                readout_mlp_irreps=readout_mlp_irreps,
                                hidden_irreps=hidden_irreps,
                                r_max=r_max,
                                num_interactions=num_interactions,
                                num_species=int(n_features),
                                graph_type=graph_type,
                                avg_num_neighbors=avg_num_neighbors,
        )
    else:
        raise NotImplementedError
    
    print("--------------------------------------------")
    print("Num. params:")
    params = net.init(jax.random.PRNGKey(0), 
                      jnp.zeros((1, n_frames*dim)), 
                      jnp.zeros((1,)), 
                      jnp.zeros((1, n_frames*n_features), dtype=int)
            )
    print(sum(x.size for x in jax.tree_leaves(params)))
    print("--------------------------------------------")

    cnf = FlowMatchingCNF(init=net.init, apply=net.apply, get_x_t_and_conditional_u_t=get_cond_vector_field,
                          sample_base=base._sample_n, sample_and_log_prob_base=base.sample_and_log_prob,
                          log_prob_base=base.log_prob)

    return cnf

"""Build's CNF for application to Cartesian coordinates of molecules."""
from typing import Sequence

from functools import partial

import jax.numpy as jnp
import distrax

from ecnf.cnf.core import FlowMatchingCNF, optimal_transport_conditional_vf
from ecnf.cnf.zero_com_base import FlatZeroCoMGaussian
from ecnf.cnf.flat_models import FlatEGNN, FlatMACE


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
        num_species: int
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
                       output_irreps="0e + 1o",
                       readout_mlp_irreps="16x0e + 16x1o",
                       hidden_irreps="256x0e + 256x1o",
                       r_max=5.0,  # Angstroms ?
                       num_interactions=2,
                       epsilon=0.4,
                       train_graphs=None,        # how to get this ?
                       num_species=num_species,  # TODO: what to provide here ?
                       graph_type="fc",          # "fc"/"nbh"/"mace" (TODO: implement "mace")
                       avg_num_neighbors=None    # TODO: get train_graphs and set to "average"  
        )

    else:
        raise NotImplementedError

    cnf = FlowMatchingCNF(init=net.init, apply=net.apply, get_x_t_and_conditional_u_t=get_cond_vector_field,
                          sample_base=base._sample_n, sample_and_log_prob_base=base.sample_and_log_prob,
                          log_prob_base=base.log_prob)

    return cnf

import chex
import jax.random
import jax.numpy as jnp

import sys
sys.path.append("./")
from ecnf.nets.mace_diffusion.macegnn import MACEDiffusionAdapted
from ecnf.utils.test import assert_function_is_equivariant


if __name__ == '__main__':
    n_nodes = 5
    dim = 3
    readout_mlp_irreps = "16x0e + 16x1o"
    hidden_irreps = "256x0e + 256x1o"
    r_max = 5.0
    num_interactions = 2
    num_species = 1
    n_nodes = 4
    graph_type = "fc"
    avg_num_neighbors = None

    key = jax.random.PRNGKey(0)

    net = MACEDiffusionAdapted(dim=dim,
                               MLP_irreps=readout_mlp_irreps,
                               hidden_irreps=hidden_irreps,
                               r_max=r_max,
                               num_interactions=num_interactions,
                               num_species=num_species,
                               n_nodes=n_nodes,
                               graph_type=graph_type,
                               avg_num_neighbors=avg_num_neighbors,
                               )

    dummy_pos = jnp.ones((n_nodes, dim))
    # dummy_feat = jnp.ones((n_nodes, 2))
    dummy_feat = jnp.ones(n_nodes, dtype=jnp.int32)
    dummy_time_embed = jnp.ones(11)

    params = net.init(key, dummy_pos, dummy_feat, dummy_time_embed)

    def eq_fn(pos: chex.Array) -> chex.Array:
        return net.apply(params, pos, dummy_feat, dummy_time_embed)


    assert_function_is_equivariant(equivariant_fn=eq_fn, n_nodes=n_nodes, dim=dim)

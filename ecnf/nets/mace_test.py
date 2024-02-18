import chex
import jax.random
import jax.numpy as jnp

import sys
sys.path.append("./")
from ecnf.nets.macegnn_adapted import MACEAdapted
from ecnf.utils.test import assert_function_is_rotation_equivariant, assert_function_is_translation_invariant, assert_function_is_translation_equivariant


if __name__ == '__main__':
    n_nodes = 5
    dim = 3
    readout_mlp_irreps = "16x0e + 16x1o"
    hidden_irreps = "256x0e + 256x1o"
    r_max = 5.0
    num_interactions = 2
    num_species = 1
    graph_type = "fc"
    avg_num_neighbors = None
    max_ell = 3

    net = MACEAdapted(dim=dim,
                      readout_mlp_irreps=readout_mlp_irreps,
                      hidden_irreps=hidden_irreps,
                      r_max=r_max,
                      num_interactions=num_interactions,
                      num_species=num_species,
                      n_nodes=n_nodes,
                      graph_type=graph_type,
                      avg_num_neighbors=avg_num_neighbors,
                      max_ell=max_ell,
                    )

    key = jax.random.PRNGKey(0)
    dummy_pos = jax.random.normal(key, (n_nodes, dim))
    dummy_feat = jnp.ones((n_nodes,), dtype=jnp.int32)
    dummy_time_embed = jax.random.normal(key, (11,))

    params = net.init(key, dummy_pos, dummy_feat, dummy_time_embed)

    def eq_fn(pos: chex.Array) -> chex.Array:
        return net.apply(params, pos, dummy_feat, dummy_time_embed)

    try:
        assert_function_is_rotation_equivariant(equivariant_fn=eq_fn, n_nodes=n_nodes, dim=dim)
        print("Rotation equivariance test passed")
    except AssertionError:
        print("Rotation equivariance test failed")

    try:
        assert_function_is_translation_invariant(equivariant_fn=eq_fn, n_nodes=n_nodes, dim=dim)
        print("Translation invariance test passed")
    except AssertionError:
        print("Translation invariance test failed")

    try:
        assert_function_is_translation_equivariant(equivariant_fn=eq_fn, n_nodes=n_nodes, dim=dim)
        print("Translation equivariance test passed")
    except AssertionError:
        print("Translation equivariance test failed")
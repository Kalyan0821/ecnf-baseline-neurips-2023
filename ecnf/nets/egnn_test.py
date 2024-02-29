import chex
import jax.random
import jax.numpy as jnp

import sys
sys.path.append("./")
from ecnf.nets.egnn import EGNN
from ecnf.utils.test import assert_function_is_rotation_equivariant, assert_function_is_translation_invariant, assert_function_is_translation_equivariant


if __name__ == '__main__':
    n_nodes = 5
    dim = 3

    egnn = EGNN(
        name='dogfish',
        n_blocks=2,
        mlp_units=(16, 16),
        n_invariant_feat_hidden=32,
    )

    key = jax.random.PRNGKey(0)
    dummy_pos = jax.random.normal(key, (n_nodes, dim)) * 10
    dummy_feat = jnp.ones((n_nodes, 1), dtype=jnp.int32)
    dummy_time_embed = jax.random.normal(key, (11,)) * 10

    params = egnn.init(key, dummy_pos, dummy_feat, dummy_time_embed)

    def eq_fn(pos: chex.Array) -> chex.Array:
        return egnn.apply(params, pos, dummy_feat, dummy_time_embed)

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
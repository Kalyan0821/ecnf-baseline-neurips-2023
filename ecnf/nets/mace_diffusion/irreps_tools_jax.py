from typing import List, Tuple
import e3nn_jax as e3nn
import flax.linen as nn
import chex
import jax.numpy as jnp


def tp_out_irreps_with_instructions(
    irreps1: e3nn.Irreps, 
    irreps2: e3nn.Irreps, 
    target_irreps: e3nn.Irreps
) -> Tuple[e3nn.Irreps, List]:

    irreps1 = e3nn.Irreps(irreps1)
    irreps2 = e3nn.Irreps(irreps2)
    target_irreps = e3nn.Irreps(target_irreps)

    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, e3nn.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = e3nn.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    return irreps_out, instructions


def reshape_irreps(tensor: chex.Array, irreps: e3nn.Irreps) -> chex.Array:
    irreps = e3nn.Irreps(irreps)
    
    ix = 0
    out = []
    batch, _ = tensor.shape
    for mul, ir in irreps:
        d = ir.dim
        field = tensor[:, ix : ix + mul * d]  # (batch=n_nodes, mul * repr)
        ix += mul * d
        field_array = field.array.reshape((batch, mul, d))
        field_IrrepsArray = e3nn.IrrepsArray(ir, field_array)
        out.append(field_IrrepsArray)

    return e3nn.concatenate(out, axis=-1)
from typing import List, Tuple
import e3nn_jax as e3nn
import flax.linen as nn
import chex
import jax.numpy as jnp

# Based on mir-group/nequip
def tp_out_irreps_with_instructions(
    irreps1: e3nn.Irreps, 
    irreps2: e3nn.Irreps, 
    target_irreps: e3nn.Irreps
) -> Tuple[e3nn.Irreps, List]:
    
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


class reshape_irreps(nn.Module):
    irreps: e3nn.Irreps

    def __call__(self, tensor: chex.Array) -> chex.Array:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, ir in self.irreps:
            d = ir.dim
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return jnp.concatenate(out, axis=-1)
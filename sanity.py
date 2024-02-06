from functools import partial
from e3nn import o3
import e3nn_jax as e3nn
import torch
import jax.numpy as jnp
import haiku as hk
import jax
from flax import linen as nn

from ecnf.nets.mace_diffusion.symmetric_contraction import SymmetricContraction as SymmetricContractionDiffTorch
from ecnf.nets.mace_modules.symmetric_contraction import SymmetricContraction as SymmetricContractionMACEJax


a = "1x0e+1x1o+1x2e+1x3o"
b = "256x0e+64x1o"

print(e3nn.Irreps(a).dim)
print(e3nn.Irreps(b).dim)

x = torch.randn((75, 19, e3nn.Irreps(a).dim))
y = torch.ones((75, 19))


try:
    z = SymmetricContractionDiffTorch(irreps_in=o3.Irreps(a), 
                                      irreps_out=o3.Irreps(b), 
                                      correlation=3, 
                                      num_elements=1)(x, y)
    print(z.shape)
except:
    print("Error")

print("--------------------------------------------------")


@hk.transform
@partial(hk.vmap, in_axes=(0, 0), split_rng=False)
def scmacejax(x, y):
    model = SymmetricContractionMACEJax(correlation=3,
                                        keep_irrep_out=e3nn.Irreps(b),
                                        num_species=1)
    print("In, pre-reshape", x.shape, y.shape)
    x = x.mul_to_axis().remove_nones()

    print("In", x.shape, y.shape)
    z = model(x, y)
    print("Out", z.shape)
    
    z = z.axis_to_mul()
    print("Out, post-reshape", z.shape)

    return z


params = scmacejax.init(jax.random.PRNGKey(0), 
                        e3nn.IrrepsArray(a, jnp.zeros(x.shape)), 
                        jnp.zeros(y.shape, dtype=jnp.int32), 
                )
print("---------------")
z = scmacejax.apply(params, 
                    jax.random.PRNGKey(42),
                    e3nn.IrrepsArray(a, jnp.array(x)), 
                    jnp.array(y, dtype=jnp.int32),
                )
print("---------------")
print(z.shape)


exit()


class Linear(nn.Module):
    input_dim: int
    hidden_dim: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.xavier_uniform(), (self.input_dim, self.hidden_dim))
        self.bias = self.param('bias', nn.initializers.zeros, (self.hidden_dim,))
        self.down = nn.Dense(features=1)

    def __call__(self, x):
        x = jnp.dot(x, self.weight) + self.bias
        x = self.down(x)
        return x
    
model = Linear(128, 64)

input_tensor = jax.random.normal(jax.random.PRNGKey(42), (17, 128))

params = model.init(jax.random.PRNGKey(0), input_tensor)
for p in params["params"]:
    print(p, params["params"][p])

# Perform a forward pass
output = model.apply(params, input_tensor)

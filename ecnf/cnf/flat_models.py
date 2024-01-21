from flax import linen as nn
import chex
from ecnf.nets.egnn import EGNN
import jax.numpy as jnp
from typing import Any, Sequence


def get_timestep_embedding(timesteps: chex.Array, embedding_dim: int):
    """Build sinusoidal embeddings (from Fairseq)."""
    # https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb#scrollTo=O5rq6xovwhgP

    assert timesteps.ndim == 1
    timesteps = timesteps * 1000

    half_dim = embedding_dim // 2
    emb = jnp.log(10_000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def reshape_and_embed(positions, node_features, time,
                      n_nodes, dim, n_features, n_invariant_feat_hidden, time_embedding_dim):
    chex.assert_rank(positions, 2)
    chex.assert_rank(node_features, 2)
    chex.assert_rank(time, 1)

    positions = jnp.reshape(positions, (positions.shape[0], n_nodes, dim))
    node_features = jnp.reshape(node_features, (node_features.shape[0], n_nodes, -1))
    node_features = nn.Embed(num_embeddings=n_features, features=n_invariant_feat_hidden)(
        jnp.squeeze(node_features, axis=-1))
    time_embedding = get_timestep_embedding(time, time_embedding_dim)

    return positions, node_features, time_embedding


class FlatEGNN(nn.Module):
    n_nodes: int
    dim: int
    n_features: int
    n_invariant_feat_hidden: int
    time_embedding_dim: int
    n_blocks_egnn: int
    mlp_units: Sequence[int]

    @nn.compact
    def __call__(self,
                 positions: chex.Array,     # (B, n_nodes*dim)
                 time: chex.Array,          # (B,)
                 node_features: chex.Array  # (B, n_nodes*n_features)
                 ) -> chex.Array:

        (positions, node_features, time_embedding) = reshape_and_embed(positions, node_features, time,
                                                                       self.n_nodes, self.dim, self.n_features, self.n_invariant_feat_hidden, self.time_embedding_dim)

        net = EGNN(n_blocks=self.n_blocks_egnn,
                   mlp_units=self.mlp_units,
                   n_invariant_feat_hidden=self.n_invariant_feat_hidden)

        vectors = net(positions,      # (B, n_nodes, dim) 
                      node_features,  # (B, n_nodes, n_invariant_feat_hidden) 
                      time_embedding  # (B, time_embedding_dim)
                      )  # (B, n_nodes, dim)
        
        flat_vectors = jnp.reshape(vectors, (vectors.shape[0], self.n_nodes*self.dim))
        return flat_vectors  # (B, n_nodes*dim)
    

class FlatMACE(nn.Module):

    @nn.compact
    def __call__(self,
                 positions: chex.Array,     # (B, n_nodes*dim)
                 time: chex.Array,          # (B,)
                 node_features: chex.Array  # (B, n_nodes*n_features)
                 ) -> chex.Array:

        (positions, node_features, time_embedding) = reshape_and_embed(positions, node_features, time,
                                                                       self.n_nodes, self.dim, self.n_features, self.n_invariant_feat_hidden, self.time_embedding_dim)

        # [l=0,p=e] (time), [l=1,p=o] (position)
        #  1x0e,             1x1o
        
        raise NotImplementedError
        
        net = EGNN(n_blocks=self.n_blocks_egnn,
                   mlp_units=self.mlp_units,
                   n_invariant_feat_hidden=self.n_invariant_feat_hidden)

        vectors = net(positions,      # (B, n_nodes, dim) 
                      node_features,  # (B, n_nodes, n_invariant_feat_hidden) 
                      time_embedding  # (B, time_embedding_dim)
                      )  # (B, n_nodes, dim)
        
        flat_vectors = jnp.reshape(vectors, (vectors.shape[0], self.n_nodes*self.dim))
        return flat_vectors  # (B, n_nodes*dim)







    









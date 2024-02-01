from flax import linen as nn
import chex
import jax.numpy as jnp
from typing import Sequence, List
import e3nn_jax as e3nn
from ecnf.nets.egnn import EGNN
from ecnf.nets.macegnn_adapted import MACEAdapted
from ecnf.nets.mace_diffusion.macegnn import MACEDiffusionAdapted
import jraph


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
                      n_nodes, dim, n_features, n_invariant_feat_hidden, time_embedding_dim,
                      skip_node_features=False):
    chex.assert_rank(positions, 2)
    chex.assert_rank(time, 1)

    positions = jnp.reshape(positions, (positions.shape[0], n_nodes, dim))

    if not skip_node_features:
        chex.assert_rank(node_features, 2)
        node_features = jnp.reshape(node_features, (node_features.shape[0], n_nodes, -1))  # (B, n_nodes, n_features)
        node_features = nn.Embed(num_embeddings=n_features, features=n_invariant_feat_hidden)(jnp.squeeze(node_features, axis=-1))
        
    time_embedding = get_timestep_embedding(time, time_embedding_dim)

    return positions, node_features, time_embedding


class FlatEGNN(nn.Module):
    n_nodes: int
    dim: int
    n_features: int
    n_invariant_feat_hidden: int
    time_embedding_dim: int
    # egnn specific
    n_blocks_egnn: int
    mlp_units: Sequence[int]

    @nn.compact
    def __call__(self,
                 positions: chex.Array,     # (B, n_nodes*dim)
                 time: chex.Array,          # (B,)
                 node_features: chex.Array  # (B, n_nodes*n_features)
                 ) -> chex.Array:
        
        # (B, n_nodes, dim), (B, n_nodes, n_invariant_feat_hidden), (B, time_embedding_dim)
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
    n_nodes: int
    dim: int
    n_features: int
    time_embedding_dim: int
    # mace specific
    output_irreps: e3nn.Irreps
    readout_mlp_irreps: e3nn.Irreps
    hidden_irreps: e3nn.Irreps
    r_max: float
    num_interactions: int
    epsilon: float
    train_graphs: List[jraph.GraphsTuple]
    num_species: int
    graph_type: str
    avg_num_neighbors: float
    output_mode: str

    @nn.compact
    def __call__(self,
                 positions: chex.Array,     # (B, n_nodes*dim)
                 time: chex.Array,          # (B,)
                 node_features: chex.Array  # (B, n_nodes*n_features)
                 ) -> chex.Array:

        assert node_features.shape[-1] == self.n_nodes
        
        # (B, n_nodes, dim), _, (B, time_embedding_dim)
        (positions, _, time_embedding) = reshape_and_embed(positions, None, time,
                                                           self.n_nodes, self.dim, None, None, self.time_embedding_dim,
                                                           skip_node_features=True)
        
        net = MACEAdapted(dim=self.dim,
                          output_irreps=self.output_irreps,
                          readout_mlp_irreps=self.readout_mlp_irreps,
                          hidden_irreps=self.hidden_irreps,
                          r_max=self.r_max,
                          num_interactions=self.num_interactions,
                          epsilon=self.epsilon,
                          train_graphs=self.train_graphs,
                          num_species=self.num_species,
                          n_nodes=self.n_nodes,
                          graph_type=self.graph_type,
                          avg_num_neighbors=self.avg_num_neighbors,
                          output_mode=self.output_mode)
                                
        vectors = net(positions,      # (B, n_nodes, dim) 
                      node_features,  # (B, n_nodes) 
                      time_embedding  # (B, time_embedding_dim)
                      )  # (B, n_nodes, dim)
        
        flat_vectors = jnp.reshape(vectors, (vectors.shape[0], self.n_nodes*self.dim))
        return flat_vectors  # (B, n_nodes*dim)


class FlatMACEDiffusion(nn.Module):
    n_nodes: int
    dim: int
    n_features: int
    time_embedding_dim: int
    # mace specific
    readout_mlp_irreps: e3nn.Irreps
    hidden_irreps: e3nn.Irreps
    r_max: float
    num_interactions: int
    num_species: int
    graph_type: str
    avg_num_neighbors: float

    @nn.compact
    def __call__(self,
                 positions: chex.Array,     # (B, n_nodes*dim)
                 time: chex.Array,          # (B,)
                 node_features: chex.Array  # (B, n_nodes*n_features)
                 ) -> chex.Array:

        assert node_features.shape[-1] == self.n_nodes
        
        # (B, n_nodes, dim), _, (B, time_embedding_dim)
        (positions, _, time_embedding) = reshape_and_embed(positions, None, time,
                                                           self.n_nodes, self.dim, None, None, self.time_embedding_dim,
                                                           skip_node_features=True)
        
        net = MACEDiffusionAdapted(dim=self.dim,
                                   readout_mlp_irreps=self.readout_mlp_irreps,
                                   hidden_irreps=self.hidden_irreps,
                                   r_max=self.r_max,
                                   num_interactions=self.num_interactions,
                                   num_species=self.num_species,
                                   n_nodes=self.n_nodes,
                                   graph_type=self.graph_type,
                                   avg_num_neighbors=self.avg_num_neighbors,
                                   time_embedding_dim=self.time_embedding_dim)
                                
        vectors = net(positions,      # (B, n_nodes, dim) 
                      node_features,  # (B, n_nodes) 
                      time_embedding  # (B, time_embedding_dim)
                      )  # (B, n_nodes, dim)
        
        flat_vectors = jnp.reshape(vectors, (vectors.shape[0], self.n_nodes*self.dim))
        return flat_vectors  # (B, n_nodes*dim)




    









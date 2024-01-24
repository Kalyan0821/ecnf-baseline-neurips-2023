from typing import Callable, Sequence, Tuple, List
import jraph

import jax.numpy as jnp
import jax
import e3nn_jax as e3nn
import chex
from flax import linen as nn

from ecnf.utils.graph import get_senders_and_receivers_fully_connected
from ecnf.utils.numerical import safe_norm
from ecnf.nets.mlp import StableMLP, MLP
from ecnf.nets.mace_tools.gin_model import model as mace_model
from ecnf.nets.mace_tools.utils import get_edge_relative_vectors
from ecnf.nets.mace_data.utils import graph_from_configuration, load_from_xyz
from ecnf.nets.mace_data.neighborhood import get_neighborhood
import haiku.experimental.flax as hkflax


class MACEGNN(nn.Module):
    """Configuration of MACEGNN."""
    dim: int
    output_irreps: e3nn.Irreps
    readout_mlp_irreps: e3nn.Irreps
    hidden_irreps: e3nn.Irreps
    r_max: float
    num_interactions: int
    epsilon: float
    train_graphs: List[jraph.GraphsTuple]
    num_species: int
    n_nodes: int
    graph_type: str
    avg_num_neighbors: float

    def setup(self):
        self.mace_model = mace_model(dim=self.dim,
                                     output_irreps=self.output_irreps,
                                     readout_mlp_irreps=self.readout_mlp_irreps,
                                     hidden_irreps=self.hidden_irreps,
                                     r_max=self.r_max,
                                     num_interactions=self.num_interactions,
                                     epsilon=self.epsilon,
                                     train_graphs=self.train_graphs,
                                     num_species=self.num_species,
                                     n_nodes=self.n_nodes,
                                     avg_num_neighbors=self.avg_num_neighbors)
        
        self.mace_model_flax = hkflax.Module(self.mace_model)

    @nn.compact
    def __call__(self,
        positions: chex.Array,        # (B, n_nodes, dim) 
        node_features: chex.Array,    # (B, n_nodes)
        global_features: chex.Array,  # (B, time_embedding_dim)
    ) -> chex.Array:
        assert positions.ndim in (2, 3)
        vmap = positions.ndim == 3
        if vmap:
            return jax.vmap(self.call_single)(positions, node_features, global_features)
        else:
            return self.call_single(positions, node_features, global_features)


    def call_single(self,
        positions: chex.Array,        # (n_nodes, dim)
        node_features: chex.Array,    # (n_nodes,)
        global_features: chex.Array,  # (time_embedding_dim,)
    ) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(positions, 2)
        chex.assert_rank(node_features, 1)
        chex.assert_rank(global_features, 1)
        chex.assert_axis_dimension(node_features, 0, self.n_nodes)

        if self.graph_type == "fc":
            senders, receivers = get_senders_and_receivers_fully_connected(self.n_nodes)
            edge_vectors = positions[receivers] - positions[senders]

        elif self.graph_type == "nbh":
            senders, receivers, _ = get_neighborhood(positions=positions, cutoff=self.r_max)
            edge_vectors = positions[receivers] - positions[senders]

        elif self.graph_type == "mace":
            # _, train_config = load_from_xyz(file_or_path=path)[0]
            # graph = graph_from_configuration(train_config, cutoff=self.r_max)
            # senders = graph.senders
            # receivers = graph.receivers
            # edge_vectors = get_edge_relative_vectors(positions=positions,
            #                                          senders=senders,
            #                                          receivers=receivers,
            #                                          shifts=graph.edges.shifts,
            #                                          cell=graph.globals.cell,
            #                                          n_edge=graph.n_edge)
            raise NotImplementedError

        else:
            raise NotImplementedError

        vectors = self.mace_model_flax(edge_vectors, node_features, senders, receivers, global_features)

        chex.assert_shape(vectors, (self.n_nodes, self.dim))

        vectors = vectors - positions

        vectors = vectors * self.param("final_scaling", nn.initializers.ones_init(), ())

        return vectors




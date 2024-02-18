from typing import Tuple, List
import jraph

import jax
import e3nn_jax as e3nn
import chex
from flax import linen as nn

from ecnf.utils.graph import get_graph_inputs
from ecnf.nets.mace_tools.gin_model import model as mace_model
import haiku.experimental.flax as hkflax


class MACEAdapted(nn.Module):
    """This implementation just tries to modify the energy predictor (force-field) model appropriately."""
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
    output_mode: str

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
                                     avg_num_neighbors=self.avg_num_neighbors,
                                     output_mode=self.output_mode)
        self.mace_model = hkflax.Module(self.mace_model)  # convert to flax module

        self.final_scaling = self.param("final_scaling", nn.initializers.ones_init(), ())


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
        ):
        chex.assert_rank(positions, 2)
        chex.assert_rank(node_features, 1)
        chex.assert_rank(global_features, 1)
        chex.assert_axis_dimension(node_features, 0, self.n_nodes)

        edge_vectors, senders, receivers = get_graph_inputs(self.graph_type, positions, self.n_nodes, self.r_max)

        vectors = self.mace_model(edge_vectors=edge_vectors, 
                                  node_z=node_features, 
                                  senders=senders, 
                                  receivers=receivers,
                                  time_embedding=global_features)
        chex.assert_shape(vectors, (self.n_nodes, self.dim))

        vectors = vectors * self.final_scaling

        return vectors




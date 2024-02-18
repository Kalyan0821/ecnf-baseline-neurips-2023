from typing import Tuple
import chex
import jax.numpy as jnp

from ecnf.nets.mace_data.neighborhood import get_neighborhood
from ecnf.nets.mace_data.utils import graph_from_configuration, load_from_xyz
from ecnf.nets.mace_tools.utils import get_edge_relative_vectors


def get_senders_and_receivers_fully_connected(n_nodes: int) -> Tuple[chex.Array, chex.Array]:
    """Get senders and receivers for fully connected graph of `n_nodes`."""
    receivers = []
    senders = []
    for i in range(n_nodes):
        for j in range(n_nodes - 1):
            receivers.append(i)
            senders.append((i + 1 + j) % n_nodes)
    return jnp.array(senders, dtype=int), jnp.array(receivers, dtype=int)


def get_graph_inputs(graph_type, positions, n_nodes, r_max, stack=False):
    if graph_type == "fc":
        senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)
        edge_vectors = positions[receivers] - positions[senders]

    elif graph_type == "nbh":
        raise NotImplementedError("Doesn't work with jax-traced positions")
        assert r_max is not None
        senders, receivers, shifts = get_neighborhood(positions=positions, cutoff=r_max)
        edge_vectors = positions[receivers] - positions[senders]

    elif graph_type == "mace":
        raise NotImplementedError
        assert r_max is not None
        _, train_config = load_from_xyz(file_or_path=path)[0]
        graph = graph_from_configuration(train_config, cutoff=r_max)
        senders = graph.senders
        receivers = graph.receivers
        edge_vectors = get_edge_relative_vectors(positions=positions,
                                                 senders=senders,
                                                 receivers=receivers,
                                                 shifts=graph.edges.shifts,  # applied on senders
                                                 cell=graph.globals.cell,
                                                 n_edge=graph.n_edge)

    else:
        raise NotImplementedError
    
    if stack:
        edge_indices = jnp.stack([senders, receivers], axis=0)
        assert edge_indices.shape[0] == 2
        
        return edge_vectors, edge_indices

    return edge_vectors, senders, receivers
import logging
from typing import Callable, List

import ase.data
import e3nn_jax as e3nn
import gin
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np

import ecnf.nets.mace_modules as modules
import ecnf.nets.mace_tools as tools
import ecnf.nets.mace_data as data


gin.register(jax.nn.silu)
gin.register(jax.nn.relu)
gin.register(jax.nn.gelu)
gin.register(jnp.abs)
gin.register(jnp.tanh)
gin.register("identity")(lambda x: x)

gin.register("std_scaling")(tools.compute_mean_std_atomic_inter_energy)
gin.register("rms_forces_scaling")(tools.compute_mean_rms_energy_forces)


@gin.configurable
def constant_scaling(graphs, atomic_energies, *, mean=0.0, std=1.0):
    return mean, std


@gin.configurable
def bessel_basis(length, max_length, number: int = 8):
    return e3nn.bessel(length, number, max_length)


@gin.configurable
def soft_envelope(
    length, max_length, arg_multiplicator: float = 2.0, value_at_origin: float = 1.2
):
    return e3nn.soft_envelope(
        length,
        max_length,
        arg_multiplicator=arg_multiplicator,
        value_at_origin=value_at_origin,
    )


@gin.configurable
def polynomial_envelope(length, max_length, degree0: int, degree1: int):
    return e3nn.poly_envelope(degree0, degree1, max_length)(length)


@gin.configurable
def u_envelope(length, max_length, p: int):
    return e3nn.poly_envelope(p - 1, 2, max_length)(length)


gin.external_configurable(modules.LinearNodeEmbeddingBlock, "LinearEmbedding")


@gin.configurable
class LinearMassEmbedding(hk.Module):
    def __init__(self, num_species: int, irreps_out: e3nn.Irreps):
        super().__init__()
        self.num_species = num_species
        self.irreps_out = e3nn.Irreps(irreps_out).filter("0e").regroup()

    def __call__(self, node_specie: jnp.ndarray) -> e3nn.IrrepsArray:
        w = hk.get_parameter(
            "embeddings",
            shape=(self.num_species, self.irreps_out.dim),
            dtype=jnp.float32,
            init=hk.initializers.RandomNormal(),
        )
        atomic_masses = jnp.asarray(ase.data.atomic_masses)[node_specie] / 90.0  # [...]
        return e3nn.IrrepsArray(
            self.irreps_out, w[node_specie] * atomic_masses[..., None]
        )


@gin.configurable
def model(
    *,
    output_irreps: e3nn.Irreps = None,  # "1o", or "0e + 1o"
    dim: int = 3,  # 2 or 3
    r_max: float,
    train_graphs: List[jraph.GraphsTuple] = None,
    num_interactions=2,
    num_species: int = None,

    avg_num_neighbors: float = "average",
    avg_r_min: float = None,
    path_normalization="path",
    gradient_normalization="path",
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray] = bessel_basis,
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray] = soft_envelope,
    **kwargs,
):
    assert output_irreps in ["1o", "0e + 1o"]
    assert dim in [2, 3]
    
    if train_graphs is None:
        z_table = None
    else:
        z_table = data.get_atomic_number_table_from_zs(
            z for graph in train_graphs for z in graph.nodes.species
        )
    logging.info(f"z_table= {z_table}")

    if avg_num_neighbors == "average":
        avg_num_neighbors = tools.compute_avg_num_neighbors(train_graphs)
        logging.info(
            f"Compute the average number of neighbors: {avg_num_neighbors:.3f}"
        )
    else:
        logging.info(f"Use the average number of neighbors: {avg_num_neighbors:.3f}")

    if avg_r_min == "average":
        avg_r_min = tools.compute_avg_min_neighbor_distance(train_graphs)
        logging.info(f"Compute the average min neighbor distance: {avg_r_min:.3f}")
    elif avg_r_min is None:
        logging.info("Do not normalize the radial basis (avg_r_min=None)")
    else:
        logging.info(f"Use the average min neighbor distance: {avg_r_min:.3f}")

    # check that num_species is consistent with the dataset
    if z_table is None:
        if train_graphs is not None:
            for graph in train_graphs:
                if not np.all(graph.nodes.species < num_species):
                    raise ValueError(
                        f"max(graph.nodes.species)={np.max(graph.nodes.species)} >= num_species={num_species}"
                    )
    else:
        if max(z_table.zs) >= num_species:
            raise ValueError(
                f"max(z_table.zs)={max(z_table.zs)} >= num_species={num_species}"
            )

    kwargs.update(
        dict(
            r_max=r_max,
            avg_num_neighbors=avg_num_neighbors,
            num_interactions=num_interactions,
            avg_r_min=avg_r_min,
            num_species=num_species,
            radial_basis=radial_basis,
            radial_envelope=radial_envelope,
        )
    )
    logging.info(f"Create MACE with parameters {kwargs}")

    @hk.without_apply_rng
    @hk.transform
    def model_(
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_z: jnp.ndarray,  # [n_nodes]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> jnp.ndarray:
        e3nn.config("path_normalization", path_normalization)
        e3nn.config("gradient_normalization", gradient_normalization)

        mace = modules.MACE(output_irreps=output_irreps, **kwargs)

        if hk.running_init():
            logging.info(
                "model: "
                f"num_features={mace.num_features} "
                f"hidden_irreps={mace.hidden_irreps} "
                f"interaction_irreps={mace.interaction_irreps} ",
            )

        contributions = mace(
            vectors, node_z, senders, receivers
        )  # [n_nodes, num_interactions, output_irreps.dim]
        node_contributions = jnp.sum(contributions.array, axis=1)  # [n_nodes, output_irreps.dim]
        assert node_contributions.shape == (len(node_z), output_irreps.dim)

        # node_energies = node_contributions[:, 0]
        node_outs = node_contributions[:, -dim:]  # [n_nodes, dim]
        return node_outs


    return model_

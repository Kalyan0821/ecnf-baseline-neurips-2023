from typing import Tuple, Callable, Optional

import os
import pathlib
from functools import partial

import jax.numpy as jnp
import jax
import chex
import optax
import wandb
import chex
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
from omegaconf import DictConfig

from ecnf.targets.data import FullGraphSample
from ecnf.cnf.build_cnf import build_cnf
from ecnf.cnf.sample_and_log_prob import sample_cnf, get_log_prob
from ecnf.cnf.gradient_step import TrainingState, flow_matching_update_fn
from ecnf.utils.loop import TrainConfig
from ecnf.utils.setup_train_objects import setup_logger
from ecnf.utils.loggers import WandbLogger
from ecnf.utils.plotting import bin_samples_by_dist, get_pairwise_distances_for_plotting, get_counts
from ecnf.utils.evaluation import eval_fn, calculate_forward_ess


def setup_training(
        cfg: DictConfig,
        load_dataset: Callable[[int, int], Tuple[FullGraphSample, FullGraphSample]],
        target_log_prob_fn: Optional[Callable[[chex.Array], chex.Array]] = None
) -> TrainConfig:
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    n_samples_plotting = cfg.training.plot_batch_size

    logger = setup_logger(cfg)

    if isinstance(logger, WandbLogger) and cfg.training.save_in_wandb_dir:
        save_path = os.path.join(wandb.run.dir, cfg.training.save_dir)
    else:
        save_path = cfg.training.save_dir

    pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)


    train_data_, test_data_ = load_dataset(cfg.training.train_set_size, cfg.training.train_set_size)
    optimizer = optax.adamw(lr)

    _, unravel_pytree = ravel_pytree(train_data_[0])
    ravel_pytree_batched = jax.vmap(lambda x: ravel_pytree(x)[0])
    train_pos_flat = ravel_pytree_batched(train_data_.positions)
    train_features_flat = ravel_pytree_batched(train_data_.features)
    test_pos_flat = ravel_pytree_batched(test_data_.positions)
    test_features_flat = ravel_pytree_batched(test_data_.features)

    n_nodes, dim = train_data_.positions.shape[1:]

    cnf = build_cnf(dim=dim,
                    n_frames=n_nodes,
                    sigma_min=cfg.flow.sigma_min,
                    base_scale=cfg.flow.base_scale,
                    n_blocks_egnn=cfg.flow.network.n_blocks_egnn,
                    mlp_units=cfg.flow.network.mlp_units,
                    n_invariant_feat_hidden=cfg.flow.network.n_invariant_feat_hidden,
                    time_embedding_dim=cfg.flow.network.time_embedding_dim
                    )


    def init_state(key: chex.PRNGKey) -> TrainingState:
        params = cnf.init(key, train_pos_flat[:2], jnp.zeros(2), train_features_flat[:2])
        opt_state = optimizer.init(params=params)
        state = TrainingState(params=params, opt_state=opt_state, key=key)
        return state

    ds_size = train_pos_flat.shape[0]

    def run_epoch(state: TrainingState) -> Tuple[TrainingState, dict]:
        key, subkey = jax.random.split(state.key)
        ds_indices = jax.random.permutation(subkey, jnp.arange(ds_size))
        state = state._replace(key=key)
        infos = []

        for j in range(ds_size // batch_size):
            batch_pos = train_pos_flat[ds_indices[j*batch_size:(j+1)*batch_size]]
            batch_feat = train_features_flat[ds_indices[j*batch_size:(j+1)*batch_size]]
            state, info = flow_matching_update_fn(
                cnf=cnf,
                opt_update=optimizer.update,
                state=state,
                x_data=batch_pos,
                features=batch_feat
            )
            infos.append(info)

        info = jax.tree_map(lambda *xs: jnp.stack(xs), *infos)
        return state, info


    def eval_single_fn(data: chex.ArrayTree, key: chex.PRNGKey, mask: chex.Array, state: TrainingState) -> \
            Tuple[chex.Array, dict]:
        key1, key2 = jax.random.split(key)
        test_pos_flat, test_features_flat = data
        key_batch = jax.random.split(key1, test_pos_flat.shape[0])

        if cfg.training.eval_exact_log_prob:
            log_q = jax.vmap(get_log_prob, in_axes=(None, None, 0, 0, 0))(cnf, state.params, test_pos_flat, key_batch,
                                                                                test_features_flat)
        else:
            log_q = jax.vmap(get_log_prob, in_axes=(None, None, 0, 0, 0, None))(
                cnf, state.params, test_pos_flat, key_batch, test_features_flat, True)
        info = {}
        info.update(
            test_log_lik=jnp.mean(log_q)
        )

        if target_log_prob_fn is not None:
            test_pos = jnp.reshape(test_pos_flat, (-1, n_nodes, dim))
            log_p = target_log_prob_fn(test_pos)
            log_w = log_p - log_q
        else:
            log_w = None
        return log_w, info




    def eval_and_plot(
            state: TrainingState, key: chex.PRNGKey,
            iteration_n: int, save: bool, plots_dir: str) -> dict:


        info, log_w_fwd, flat_mask = eval_fn(
            x = (test_pos_flat, test_features_flat),
            key=key,
            eval_on_test_batch_fn=partial(eval_single_fn, state=state),
            batch_size=cfg.training.eval_batch_size,
        )

        if target_log_prob_fn is not None:
            further_info = calculate_forward_ess(log_w_fwd, mask=flat_mask)
            info.update(further_info)

        key, subkey = jax.random.split(key)
        key_batch = jax.random.split(key, n_samples_plotting)
        flow_samples_flat = jax.vmap(sample_cnf, in_axes=(None, None, 0, 0))(
            cnf, state.params, key_batch, jnp.repeat(train_features_flat[0:1], n_samples_plotting, axis=0))
        flow_samples = jnp.reshape(flow_samples_flat, (n_samples_plotting, n_nodes, dim))

        # Plot samples.
        bins_x, count_list = bin_samples_by_dist([train_data_.positions[:n_samples_plotting]], max_distance=10.)
        plotting_n_nodes = train_data_.positions.shape[1]
        pairwise_distances_flow = get_pairwise_distances_for_plotting(flow_samples, plotting_n_nodes, max_distance=10.)
        counts_flow = get_counts(pairwise_distances_flow, bins_x)

        fig1, ax = plt.subplots(1, figsize=(5, 5))
        ax.stairs(count_list[0], bins_x, label="train samples", alpha=0.4, fill=True)
        ax.stairs(counts_flow, bins_x, label="flow samples", alpha=0.4, fill=True)
        ax.legend()

        figs = [fig1, ]
        for j, figure in enumerate(figs):
            if save:
                figure.savefig(
                    os.path.join(plots_dir, "plot_%03i_iter_%08i.png" % (j, iteration_n))
                )
            else:
                plt.show()
            plt.close(figure)

        return info


    train_config = TrainConfig(
        n_iteration=cfg.training.n_training_iter,
        logger=logger,
        seed=cfg.training.seed,
        n_checkpoints=cfg.training.n_checkpoints,
        n_eval=cfg.training.n_eval,
        init_state=init_state,
        update_state=run_epoch,
        eval_and_plot_fn=eval_and_plot,
        save=cfg.training.save,
        save_dir=save_path)

    return train_config

import pickle
import numpy as np
import jax
import jax.numpy as jp
import flax, optax
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import pad_shard_unpad

from src import BASE_DIR
from src.models.gcpc import TrajNet
from src.models.gcpc.configs import ModelConfig
from src.models.gcpc.prepare_traj import prepare_config_dataset, make_state
from src.common.configs import TotalConfigs, DatasetConfig, TrainConfig
from src.datasets import AntDataLoader, AntNormalizer
from src.utils.ant_viz import GoalReachingMaze
from src.utils.train_state import TrainState
from src.utils.logging_utils import Logger, compare_recons
from src.utils.context import make_rngs, save_state

import wandb
from boxprint import bprint
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


def calc_recon_loss(batch: Dict[str, jp.ndarray], recon: jp.ndarray, pad_mask):
    bs, seq_len = recon.shape[:2]
    loss_img = optax.l2_loss(batch[batch['traj_seq'], recon])
    loss_img *= pad_mask
    loss = loss_img.mean() * bs * seq_len / pad_mask.sum()
    return loss, {"loss_img": loss_img.mean(0), "loss": loss}


def calc_action_loss(actions: jp.ndarray, preds: jp.ndarray):
    return optax.l2_loss(actions, preds).mean()


def train_step(state: TrainState, batch: Dict[str, jp.ndarray], rng: jp.ndarray, config: ModelConfig, pmap_axis=None):
    rng_names = ('dropout', )
    rngs = make_rngs(rng, rng_names)
    pad_mask = batch.pop('pad')[..., jp.newaxis]

    def loss_fn(params):
        recon = state(**batch, train=True, params=params, rngs=rngs)
        loss, info = calc_recon_loss(batch, recon, pad_mask)
        return loss, info
    
    state, info = state.apply_loss_fn(loss_fn, has_aux=True, pmap_axis=pmap_axis)
    return state, info


def eval_step(state: TrainState, batch: Dict[str, jp.ndarray], rng: jp.ndarray, config: ModelConfig, pmap_axis=None):
    rng_names = ('dropout', )
    rngs = make_rngs(rng, rng_names)
    pad_mask = batch.pop('pad')[..., jp.newaxis]
    
    recon = state(**batch, train=False, params=state.params, rngs=rngs)
    loss, info = calc_recon_loss(batch, recon, pad_mask)
    return recon, info


def train_traj(state: TrainState,
               configs: TotalConfigs,
               train_step_fn: callable,
               eval_step_fn: callable,
               sample_batch_fn: callable,
               dataloader: AntDataLoader,
               render_env: GoalReachingMaze,
               logger: Logger,
               eval_batch: jp.ndarray = None,
               log_interval: int = -1,
               save_interval: int = -1,
               eval_freq: int = 0,
               **kwargs) -> TrainState:
    
    n_epochs = configs.train_config.n_epochs
    rng = jax.random.PRNGKey(0)

    normalizer: AntNormalizer = kwargs["normalizer"]
    splits: List[Tuple[int, int]] = kwargs["splits"]
    denorm_keys = ("observations", ) + ("actions", ) if configs.model_config.state_action else ()

    if eval_batch is None:
        eval_batch = sample_batch_fn(pmap=False)

    loader_size = dataloader.size // configs.train_config.batch_size
    if kwargs.get("loader_size", 0) > 0:
        loader_size //= kwargs["loader_size"]
        print("\033[031mLoader size modified!!\033[0m")

    total_steps = n_epochs * loader_size
    global_step = flax.jax_utils.unreplicate(state.step)

    for epoch in range(n_epochs):
        pbar = tqdm(range(loader_size), desc=f"Epoch {epoch + 1}/{n_epochs}", ncols=120)
        for step in pbar:
            batch = sample_batch_fn(pmap=True)
            rng, device_rng = jax.random.split(rng)

            state, info = train_step_fn(state, batch, device_rng)
            info = flax.jax_utils.unreplicate(flax.traverse_util.flatten_dict(info, sep='/'))
            loss = info['loss'].item()

            last_lr = state.opt_state.hyperparams["schedule"][0].item()
            pbar.set_postfix({"loss": loss, "lr": last_lr, "global_step": global_step})
            global_step += 1

            if log_interval > 0 and (global_step % log_interval == 0 or global_step == 1 or global_step == total_steps - 1):
                logger.log(info, global_step, include_imgs=global_step % (log_interval * 10) == 0)
                logger.log({"lr": last_lr}, global_step)

            if save_interval > 0 and global_step % save_interval == 0:
                save_state(state, configs.train_config.save_dir / "checkpoint", global_step)
            
            if eval_freq > 0 and (step + 1) % (loader_size // eval_freq) == 0:
                rng, device_rng = jax.random.split(rng)
                device_rng = shard_prng_key(device_rng)
                eval_recon, eval_info = eval_step_fn(state, eval_batch, device_rng)
                eval_recon = flax.jax_utils.unreplicate(eval_recon)
                eval_info = flax.jax_utils.unreplicate(flax.traverse_util.flatten_dict(eval_info, sep='/'))

                compare_recons(
                    logger=logger,
                    env=render_env,
                    origs=normalizer.denormalize_concat(jax.device_get(eval_batch['traj_seq']), keys=denorm_keys, splits=splits),
                    recons=normalizer.denormalize_concat(jax.device_get(eval_recon), keys=denorm_keys, splits=splits),
                    goal_dim=configs.model_config.goal_dim,
                    global_step=global_step,
                    goal_conditioned=True
                )

                logger.log(eval_info, global_step, prefix="Eval")

    return state


def main(model_def: type[TrajNet],
         env_name: str,
         seq_len: int,
         batch_size: int,
         n_epochs: int,
         log_interval: int,
         save_interval: int,
         eval_freq: int,
         use_wandb: bool = True,
         **kwargs):
    
    dataloader, configs = prepare_config_dataset(env_name, seq_len, batch_size, n_epochs=n_epochs, **kwargs)
    render_env = GoalReachingMaze(env_name)
    n_devices = jax.device_count()
    
    sample_
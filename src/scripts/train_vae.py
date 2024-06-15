import pickle
import flax.jax_utils
import numpy as np
import jax
import jax.tree_util as jtr
import jax.numpy as jp
import flax, optax
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import pad_shard_unpad

from src import BASE_DIR
from src.models.vae import VQVAE
from src.common.configs import TotalConfigs, DatasetConfig, ModelConfig, TrainConfig
from src.datasets import AntDataLoader, AntNormalizer, make_env
from src.utils.train_state import TrainState
from src.utils.ant_viz import GoalReachingAnt
from src.utils.logging_utils import Logger, compare_recons
from src.utils.context import make_rngs, save_state
from src.scripts.vae_prepare import prepare_config_dataset, prepare_states
from src.scripts.batch_samplers import vae_batch_sampler

import wandb
from boxprint import bprint
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


def calc_recon_loss(config: ModelConfig, batch: jp.ndarray, recon: jp.ndarray):
    splits = np.cumsum([config.goal_dim, 2, config.observation_dim - 2, config.action_dim])
    diff = jp.mean((batch - recon) ** 2, axis=-0)
    goal_diff, pos_diff, obs_diff, act_diff, _ = jp.split(diff, splits, axis=-1)
    
    if not config.goal_conditional:
        goal_diff = jp.zeros_like(goal_diff)
    
    goal_obs_pos_loss = jp.concatenate([goal_diff, pos_diff], axis=-1).mean() * config.pos_weight
    obs_loss = obs_diff.mean()
    act_loss = act_diff.mean() * config.action_weight
    
    term_diff = jp.abs(jax.nn.sigmoid(recon[..., -1]) > 0.5 - batch[..., -1]).mean(0)
    term_loss = optax.sigmoid_binary_cross_entropy(recon[..., -1:], batch[..., -1:]).mean()
    
    loss = (goal_obs_pos_loss + obs_loss + act_loss + term_loss) / 4.0
    info = {"goal_obs_pos_loss": goal_obs_pos_loss,
            "obs_loss": obs_loss,
            "act_loss": act_loss,
            "term_loss": term_loss,
            "trns_loss_map": diff[..., :-1],
            "term_loss_map": term_diff,
            "loss": loss}
    
    return loss, info


def train_step(state: TrainState, batch: jp.ndarray, rng: jp.ndarray, config: TotalConfigs, pmap_axis=None):
    rng_names = ('vq', 'dropout')
    rngs = make_rngs(rng, rng_names)
    
    def loss_fn(params):
        out = state(**batch, train=True, params=params, rngs=rngs, mutable=['vq_stats'])
        (recon, vq_info), updates = out
        
        reocn_loss, info = calc_recon_loss(config.model_config, batch['traj_seq'], recon)
        loss = reocn_loss + vq_info["vq_loss"]
        
        info.update({"loss": loss, **vq_info})
        return loss, (info, updates)

    state, (info, updates) = state.apply_loss_fn(loss_fn, has_aux=True, pmap_axis=pmap_axis)
    state = state.replace(extra_variables=updates)
    return state, info


def eval_step(state: TrainState, batch: jp.ndarray, rng: jp.ndarray, config: TotalConfigs, pmap_axis=None):
    rng_names = ('vq', 'dropout')
    rngs = make_rngs(rng, rng_names)
    
    out = state(**batch, train=False, rngs=rngs)
    recon, vq_info = out
    recon_loss, info = calc_recon_loss(config.model_config, batch['traj_seq'], recon)
    loss = recon_loss + vq_info["vq_loss"]
    
    recon = recon.at[..., -1].set(jax.nn.sigmoid(recon[..., -1]) > 0.5)
    
    info = jax.lax.pmean(info, axis_name=pmap_axis)
    info.update({"loss": loss, **vq_info})
    return recon, info


def train_vae(state: TrainState,
              configs: TotalConfigs,
              train_step_fn: callable,
              eval_step_fn: callable,
              sample_batch_fn: callable,
              dataloader: AntDataLoader,
              render_env: GoalReachingAnt,
              logger: Logger,
              eval_batch: jp.ndarray = None,
              log_interval: int = -1,
              save_interval: int = -1,
              eval_freq: int = 0,
              **kwargs) -> TrainState:
    
    batch_size = configs.train_config.batch_size
    n_epochs = configs.train_config.n_epochs
    rng = jax.random.PRNGKey(0)

    normalizer: AntNormalizer = kwargs["normalizer"]
    splits: List[Tuple[int, int]] = kwargs["splits"]
    denorm_keys = ("goals", "observations", "actions")

    if eval_batch is None:
        eval_batch = sample_batch_fn(pmap=False)    # pad_shard_unpad will handle the sharding

    loader_size = dataloader.size // configs.train_config.batch_size
    if kwargs.get("loader_size", 0) > 0:
        loader_size //= kwargs["loader_size"]
        print('\33[031mLoader size modified!!!\33[0m')
    
    total_steps = n_epochs * loader_size
    global_step = flax.jax_utils.unreplicate(state.step)

    for epoch in range(n_epochs):
        pbar = tqdm(range(loader_size), desc=f"Epoch[{epoch + 1}/{n_epochs}]", ncols=120)
        for step in pbar:
            batch = sample_batch_fn(pmap=False)     # pad_shard_unpad will handle the sharding
            rng, device_rng = jax.random.split(rng)
            device_rng = shard_prng_key(device_rng)

            state, info = train_step_fn(state, batch, device_rng)
            info = flax.jax_utils.unreplicate(flax.traverse_util.flatten_dict(info, sep='/'))
            # info = flax.traverse_util.flatten_dict(info, sep='/')
            loss = info['loss'].item()

            last_lr = state.opt_state.hyperparams['schedule'][0].item()
            pbar.set_postfix({"loss": loss, "lr": last_lr, "global_step": global_step})
            global_step += 1

            if log_interval > 0 and (global_step % log_interval == 0 or global_step == 1 or global_step == total_steps - 1):
                logger.log(info, global_step, include_imgs=global_step % (log_interval * 10) == 0)
                logger.log({"lr": last_lr}, global_step)
            
            if save_interval > 0 and global_step % save_interval == 0:
                name = f"checkpoint-{str(epoch).zfill(3)}-{str(global_step).zfill(4)}"
                save_state(state, configs.train_config.save_dir / name, global_step)

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
                    goal_dim=configs.model_config.goal_input_dim,
                    global_step=global_step,
                    quantized=eval_info["enc_vq/indices"][:4],
                    goal_conditioned=configs.model_config.goal_conditional
                )

                logger.log(eval_info, global_step, prefix="Eval")

    return state


def main(model_def: type[VQVAE],
         env_name: str,
         seq_len: int,
         latent_step: int,
         batch_size: int,
         n_epochs: int,
         log_interval: int,
         save_interval: int,
         eval_freq: int,
         use_wandb: bool = True,
         **kwargs):
    
    # Prepare config and dataset ========
    dataloader, configs = prepare_config_dataset(env_name, seq_len, latent_step, batch_size, n_epochs, **kwargs)
    render_env = GoalReachingAnt(env_name)
    n_devices = jax.device_count()

    # Data sampler ========
    sample_batch_fn, (normalizer, splits) = vae_batch_sampler(dataloader, batch_size, normalize=True)

    # Eval batch ========
    eval_starts = np.arange(4) * dataloader.seq_len + 21 * 1000
    eval_batch = sample_batch_fn(starts=eval_starts, pmap=False)
    eval_batch = jtr.tree_map(lambda x: np.expand_dims(x, axis=0).repeat(n_devices, axis=0), eval_batch)
    eval_batch = jtr.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), eval_batch)

    # Scheduler & States ========
    total_steps = (dataloader.size // batch_size) * n_epochs
    scheduler = optax.cosine_onecycle_schedule(transition_steps=total_steps,
                                               peak_value=configs.train_config.learning_rate,
                                               pct_start=0.15,
                                               div_factor=50.0,
                                               final_div_factor=200)
    
    state = prepare_states(model_def, configs, scheduler)
    state = flax.jax_utils.replicate(state)

    # Pmap functions ========
    axis_name = "batch"
    p_train_step = jax.pmap(partial(train_step, config=configs, pmap_axis=axis_name), axis_name=axis_name)
    p_eval_step = jax.pmap(partial(eval_step, config=configs, pmap_axis=axis_name), axis_name=axis_name)

    p_train_step = pad_shard_unpad(p_train_step, static_argnums=(0, 2), static_return=True)
    p_eval_step = pad_shard_unpad(p_eval_step, static_argnums=(0, 2), static_return=True)

    # Wandb ========
    project_root = Path(BASE_DIR["project"])
    save_dir = configs.train_config.save_dir
    exp_name = configs.train_config.exp_name
    if use_wandb:
        run = wandb.init(project=exp_name.split('-')[0], dir=project_root, name=exp_name, config=configs.get_dict())
    else:
        run = None
    logger = Logger(run)

    bprint(f"Experiment: {exp_name}\nEnvironment: {env_name}\nDevices: {n_devices}\nLogging to: {save_dir}", width=120)
    print('\n')

    # Training loop ========
    state = train_vae(state, configs, p_train_step, p_eval_step, sample_batch_fn, dataloader,
                      render_env=render_env,
                      logger=logger,
                      eval_batch=eval_batch,
                      log_interval=log_interval,
                      save_interval=save_interval,
                      eval_freq=eval_freq,
                      normalizer=normalizer,
                      splits=splits,
                      **kwargs)
    
    # Save final model ========
    save_state(state, configs.train_config.save_dir / "checkpoint-final", total_steps)
    state = flax.jax_utils.unreplicate(state)
    configs.save(configs.train_config.save_dir)
    pickle.dump({"params": state.params, **state.extra_variables}, open(save_dir / "model_params.pkl", "wb"))

    if log_interval > 0:
        logger.run.finish()


if __name__ == "__main__":
    import chex
    env_name = "antmaze-large-play-v2"
    model_def = VQVAE

    pmap = True
    log_interval = 20
    save_interval = 2000
    eval_freq = 2
    use_wandb = True
    kwargs = {
        "model": {},
        "dataset": {"goal_conditioned": True, "hierarchical_goal": False, "p_true_goal": 1.0, "p_sub_goal": 0.0},
        "train": {},
        # "loader_size": 7000
    }

    if pmap:
        main(model_def, env_name,
             seq_len=64, latent_step=4, batch_size=512 * 4, n_epochs=9,
             log_interval=log_interval, save_interval=save_interval, eval_freq=eval_freq, use_wandb=use_wandb, **kwargs)
        
    else:
        jax.config.update("jax_platform_name", "cpu")
        chex.set_n_cpu_devices(4)
        with chex.fake_pmap_and_jit():
            main(model_def, env_name,
                 seq_len=64, latent_step=4, batch_size=64, n_epochs=9,
                 log_interval=log_interval, save_interval=save_interval, eval_freq=eval_freq, use_wandb=use_wandb, **kwargs)
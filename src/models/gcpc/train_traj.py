import pickle
import flax.jax_utils
import flax.jax_utils
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
from src.models.gcpc.batch_sampler import gcpc_batch_sampler

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
from functools import partial
from typing import Dict, List, Tuple


def calc_recon_loss(batch: Dict[str, jp.ndarray], recon: jp.ndarray, pad_mask):
    bs, seq_len = recon.shape[:2]
    loss_img = optax.l2_loss(recon, batch["traj_seq"])
    loss_img *= pad_mask
    loss = loss_img.mean() * bs * seq_len / pad_mask.sum()
    return loss, {"loss_img": loss_img.mean(0), "loss": loss}


def calc_action_loss(actions: jp.ndarray, preds: jp.ndarray):
    return optax.l2_loss(actions, preds).mean()


def train_step(state: TrainState, batch: Dict[str, jp.ndarray], rng: jp.ndarray, config: ModelConfig, pmap_axis=None):
    rng_names = ('dropout', )
    rngs = make_rngs(rng, rng_names)
    pad_mask = batch.pop('pad')

    def loss_fn(params):
        recon = state(**batch, train=True, params=params, rngs=rngs)
        loss, info = calc_recon_loss(batch, recon, pad_mask)
        return loss, info
    
    state, info = state.apply_loss_fn(loss_fn, has_aux=True, pmap_axis=pmap_axis)
    return state, info


def eval_step(state: TrainState, batch: Dict[str, jp.ndarray], rng: jp.ndarray, config: ModelConfig, pmap_axis=None):
    rng_names = ('dropout', )
    rngs = make_rngs(rng, rng_names)
    pad_mask = batch.pop('pad')
    
    recon = state(**batch, train=False, params=state.params, rngs=rngs)
    loss, info = calc_recon_loss(batch, recon, pad_mask)
    return recon, info


def train_traj(state: TrainState,
               configs: TotalConfigs,
               train_step_fn: callable,
               eval_step_fn: callable,
               sample_batch_fn: callable,
               denormalize_fn: callable,
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

    # normalizer: AntNormalizer = kwargs["normalizer"]
    # splits: List[Tuple[int, int]] = kwargs["splits"]
    # denorm_keys = ("observations", ) + ("actions", ) if configs.model_config.state_action else ()

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
            batch = sample_batch_fn(pmap=False)
            rng, device_rng = jax.random.split(rng)
            device_rng = shard_prng_key(device_rng)

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
                    origs=denormalize_fn(jax.device_get(eval_batch['traj_seq'])),
                    recons=denormalize_fn(jax.device_get(eval_recon)),
                    goal_dim=configs.model_config.goal_dim,
                    global_step=global_step,
                    goal_conditioned=True
                )

                logger.log(eval_info, global_step, prefix="Eval")

    return state


def main(model_def: type[TrajNet],  # model-config
         env_name: str,     # dataset-config
         window_size: int,
         future_size: int,
         batch_size: int,   # dataset-config
         n_epochs: int,
         log_interval: int,
         save_interval: int,
         eval_freq: int,
         use_wandb: bool = True,
         **kwargs):
    
    # Prepare config and dataset ========
    dataloader, configs = prepare_config_dataset(env_name, batch_size, window_size, future_size, n_epochs=n_epochs, **kwargs)
    render_env = GoalReachingMaze(env_name)
    n_devices = jax.device_count()
    rng = jax.random.PRNGKey(0)
    
    # Data sampler and normalizer ========
    sample_batch_fn, denormalize_fn = gcpc_batch_sampler(dataloader, batch_size, 
                                                         normalize=True, model_config=configs.model_config)
    
    # Eval batch ========
    eval_starts = np.arange(4) * dataloader.seq_len + dataloader.terminal_ids[21] + 1
    eval_batch = sample_batch_fn(pmap=False, starts=eval_starts)
    
    # Scheduler & Optimizer ========
    total_steps = (dataloader.size // batch_size) * n_epochs
    scheduler = optax.constant_schedule(value=configs.train_config.learning_rate)
    
    @optax.inject_hyperparams
    def make_optim(schedule):
        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule,
                        b1=configs.train_config.betas[0],
                        b2=configs.train_config.betas[1],
                        weight_decay=configs.train_config.weight_decay)
        )
    
    tx = make_optim(scheduler)
     
    # Model and state ========
    model = model_def(configs.model_config)
    rngs = make_rngs(rng, ('dropout', ), contain_params=True)
    
    input_sample = model._input_sample()
    variables = model.init(rngs, **input_sample, train=False)
    state = TrainState.create(model, params=variables.pop('params'), tx=tx, extra_variables=variables)
    state = flax.jax_utils.replicate(state)
    
    # Pmap functions ========
    axis_name = "batch"
    p_train_step = jax.pmap(partial(train_step, config=configs.model_config, pmap_axis=axis_name), axis_name=axis_name)
    p_eval_step = jax.pmap(partial(eval_step, config=configs.model_config, pmap_axis=axis_name), axis_name=axis_name)
    
    p_train_step = pad_shard_unpad(p_train_step, static_argnums=(0, 2), static_return=True)
    p_eval_step = pad_shard_unpad(p_eval_step, static_argnums=(0, 2), static_return=True)
    
    # Logger and wandb ========
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
    
    state = train_traj(state, configs, p_train_step, p_eval_step, sample_batch_fn, denormalize_fn,
                       dataloader=dataloader,
                       render_env=render_env,
                       logger=logger,
                       eval_batch=eval_batch,
                       log_interval=log_interval,
                       save_interval=save_interval,
                       eval_freq=eval_freq)
    
    save_state(state, configs.train_config.save_dir / "checkpoint", total_steps)
    state = flax.jax_utils.unreplicate(state)
    configs.save(configs.train_config.save_dir)
    pickle.dump({"params": state.params, **state.extra_variables}, open(save_dir / "model_params.pkl", "wb"))
    
    if log_interval > 0:
        logger.run.finish()
        
        
if __name__ == "__main__":
    import chex
    
    env_name = "maze2d-large-v1"
    model_def = TrajNet
    
    log_interval = 20
    save_interval = 2000
    eval_freq = 2
    pmap=True
    use_wandb = False
    test = False
    
    loader_size = 1000 if test else 0
    batch_size = 256 if test else 512 * jax.device_count()
    
    kwargs = {
        "model": {"emb_dim": 256, "n_heads": 4, "n_enc_layers": 2, "n_dec_layers": 1, 
                  "n_slots": 4, "use_goal": True, "window_mask_rate": 0.25, "future_mask_rate": 1.0},
        "dataset": {"hierarchical_goal": False, "p_true_goal": 1.0, "p_sub_goal": 0.0},
        "train": {"exp_name": "GCPC_Traj"},
        "loader_size": loader_size
    }
    
    if pmap:
        main(model_def, env_name,
             window_size=10, future_size=70, batch_size=batch_size, n_epochs=10,
             log_interval=log_interval, save_interval=save_interval, eval_freq=eval_freq, use_wandb=use_wandb, **kwargs)
    
    else:
        with chex.fake_pmap_and_jit():
            main(model_def, env_name,
                 window_size=10, future_size=70, batch_size=batch_size, n_epochs=10,
                 log_interval=log_interval, save_interval=save_interval, eval_freq=eval_freq, use_wandb=use_wandb, **kwargs)
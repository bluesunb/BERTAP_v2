import pickle
import flax.jax_utils
import flax.jax_utils
import flax.traverse_util
import numpy as np
import jax
import jax.tree_util as jtr
import jax.numpy as jp
import flax, optax
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import pad_shard_unpad

from src import BASE_DIR
from src.models.vae import VQVAE
from src.models.transformer import BertWithHeads
from src.common.configs import TotalConfigs, DatasetConfig, ModelConfig, TrainConfig
from src.datasets import AntMLMDataLoader, AntNormalizer, make_env
from src.utils.train_state import TrainState
from src.utils.logging_utils import Logger, compare_recons, TabularLogger
from src.utils.context import make_rngs, save_state
from src.scripts.prior_prepare_copy import prepare_config_dataset, prepare_states
from src.scripts.batch_samplers import gpt_batch_sampler, GPTDataCollator

import wandb
from boxprint import bprint
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


def make_schedule_fn(train_config: TrainConfig, **schedule_kwargs) -> optax.Schedule:
    def constant_schedule(**kwargs):
        return optax.constant_schedule(train_config.learning_rate)
    
    def one_cycle(total_steps: int, **kwargs):
        return optax.cosine_onecycle_schedule(
            transition_steps=total_steps,
            peak_value=train_config.learning_rate,
            pct_start=0.3,
            div_factor=1e2,
            final_div_factor=1e3,
        )
        
    def bert_warmup(init_value: float, warmup_steps: int, **kwargs):
        def schedule_fn(step: int):
            scale = jp.minimum(jp.power(step, -0.5) , jp.power(warmup_steps, -1.5) * step)
            return scale * init_value + 1e-8
        return schedule_fn
    
    if train_config.scheduler_name == "constant":
        return constant_schedule(**schedule_kwargs)
    
    if train_config.scheduler_name == "one_cycle":
        return one_cycle(**schedule_kwargs)
    
    if train_config.scheduler_name == "bertwarmup":
        return bert_warmup(**schedule_kwargs)
    
    raise ValueError(f"Invalid scheduler name: {train_config.scheduler_name}")


def calc_loss(logits: Tuple[jp.ndarray, ...], labels: Tuple[jp.ndarray, ...], label_mask: jp.ndarray, nsp_weight: float = 1.0):
    n_labels = label_mask.sum()
    mlm_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
    
    mlm_loss = mlm_loss.sum() / n_labels
    
    return mlm_loss, {"mlm_loss": mlm_loss}
    
    
def train_step(state: TrainState, batch: Dict[str, jp.ndarray], rng: jp.ndarray, config: ModelConfig, pmap_axis: str = None):
    rng_names = ("dropout", )
    rngs = make_rngs(rng, rng_names, contain_params=False)
    
    def loss_fn(params):
        mlm_labels = batch.pop("labels")
        mlm_logits = state(**batch, train=True, params=params, rngs=rngs)
        label_mask = mlm_labels >= 0
        
        loss, info = calc_loss(mlm_logits, mlm_labels, label_mask, nsp_weight=config.nsp_weight)
        info['loss'] = loss
        return loss, info
    
    state, info = state.apply_loss_fn(loss_fn, has_aux=True, pmap_axis=pmap_axis)
    info = jax.lax.pmean(info, axis_name=pmap_axis)
    return state, info


def eval_step(state: TrainState, batch: jp.ndarray, rng: jp.ndarray, config: ModelConfig, pmap_axis: str = None):
    rng_names = ("dropout", )
    rngs = make_rngs(rng, rng_names, contain_params=False)

    mlm_labels = batch.pop("labels")
    mlm_logits = state(**batch, train=False, rngs=rngs)
    label_mask = mlm_labels > -100
    
    loss, info = calc_loss(mlm_logits, mlm_labels, label_mask, nsp_weight=config.nsp_weight)
    
    mlm_preds = jp.argmax(mlm_logits, axis=-1)
    mlm_acc = jp.mean((mlm_preds == mlm_labels) * label_mask)
    # mlm_acc = jp.mean((mlm_preds[..., -1] == mlm_labels[..., -1]) * label_mask[..., -1])
    
    metrics = {"loss": loss, "mlm_acc": mlm_acc}
    metrics = jax.lax.pmean(metrics, axis_name=pmap_axis)
    return metrics


def train_prior(state: TrainState,
                configs: TotalConfigs,
                train_step_fn: callable,
                eval_step_fn: callable,
                sample_batch_fn: callable,
                dataloader: AntMLMDataLoader,
                logger: Logger,
                eval_batch: jp.ndarray = None,
                log_interval: int = -1,
                save_interval: int = -1,
                eval_freq: int = 0,
                **kwargs) -> TrainState:
    
    n_epochs = configs.train_config.n_epochs
    rng = jax.random.PRNGKey(configs.train_config.seed)
    
    if eval_batch is None:
        eval_batch = sample_batch_fn(pmap=False, return_conditioned=True, rng=rng)
        
    loader_size = dataloader.size // configs.train_config.batch_size
    if kwargs.get("loader_size", 0) > 0:
        loader_size = kwargs["loader_size"]
        print('\33[031mLoader size modified!!!\33[0m')
        
    total_steps = n_epochs * loader_size
    global_step = flax.jax_utils.unreplicate(state.step).item()
    
    epoch_bar = tqdm(range(n_epochs), desc="Epochs", ncols=120, position=0)
    eval_logger = TabularLogger(["Step", "Eval Loss", "MLM Acc"], pbar=epoch_bar)
    for epoch in epoch_bar:
        pbar = tqdm(range(loader_size), desc=f"Epoch [{epoch + 1}/{n_epochs}]", ncols=120, position=1, leave=False)
        for step in pbar:
            batch = sample_batch_fn(pmap=False, return_conditioned=True, rng=rng)
            rng, device_rng = jax.random.split(rng)
            device_rng = shard_prng_key(rng)
            
            state, info = train_step_fn(state, batch, device_rng)
            info = flax.jax_utils.unreplicate(flax.traverse_util.flatten_dict(info, sep='/'))
            loss = info["loss"].item()
            
            last_lr = state.opt_state.hyperparams["schedule"][0].item()
            pbar.set_postfix({"loss": loss, "lr": last_lr, "global_step": global_step})
            global_step += 1
            
            if log_interval > 0 and (global_step % log_interval == 0 or global_step == 1 or global_step == total_steps - 1):
                logger.log({"lr": last_lr, **info}, global_step)
                
            if save_interval > 0 and global_step % save_interval == 0:
                # name = f"checkpoint-{str(epoch).zfill(3)}-{str(global_step).zfill(4)}"
                name = "checkpoint"
                save_state(state, configs.train_config.save_dir / name, global_step)
                
            if eval_freq > 0 and (step + 1) % (loader_size // eval_freq) == 0:
                rng, device_rng = jax.random.split(rng)
                device_rng = shard_prng_key(device_rng)
                eval_info = eval_step_fn(state, eval_batch, device_rng)
                eval_info = flax.jax_utils.unreplicate(flax.traverse_util.flatten_dict(eval_info, sep='/'))
                logger.log(eval_info, global_step, prefix="Eval")
                
                eval_logger.log(step=global_step,
                                eval_loss=eval_info["loss"].item(),
                                mlm_acc=eval_info["mlm_acc"].item())
                
    return state


def main(model_def: type[VQVAE],
         vae_path: Path,
         batch_size: int,
         n_epochs: int,
         log_interval: int,
         save_interval: int,
         eval_freq: int,
         use_wandb: bool = True,
         **kwargs):
    
    # Prepare config and dataset =======
    dataloader, configs, vae_params, vae_configs = prepare_config_dataset(vae_path, batch_size, n_epochs, **kwargs)

    n_devices = jax.local_device_count()
    
    # Data collator =======
    vae = VQVAE(vae_configs.model_config, training=False)
    data_collator = GPTDataCollator(vae, vae_params, configs.model_config, seed=configs.train_config.seed)
    
    # Data sampler =======
    sample_batch_fn, (normalizer, splits) = gpt_batch_sampler(dataloader, batch_size, data_collator, 
                                                              normalize=True, hierarchical_goal=configs.data_config.hierarchical_goal)
    
    # Eval batch =======
    eval_starts = np.arange(4) * dataloader.seq_len + 21 * 1000
    eval_batch = sample_batch_fn(starts=eval_starts, return_conditioned=True, pmap=False)

    # Schedule & States =======
    total_steps = (dataloader.size // batch_size) * n_epochs
    scheduler = make_schedule_fn(
        configs.train_config,
        total_steps=total_steps,
        init_value=configs.train_config.learning_rate * (configs.model_config.emb_dim ** -0.5),
        warmup_steps=total_steps // 10
    )
    
    state = prepare_states(model_def, configs, scheduler)
    state = flax.jax_utils.replicate(state)
    
    # Pmap functions =======
    axis_name = "batch"
    p_train_step = jax.pmap(partial(train_step, config=configs.model_config, pmap_axis=axis_name), axis_name=axis_name)
    p_eval_step = jax.pmap(partial(eval_step, config=configs.model_config, pmap_axis=axis_name), axis_name=axis_name)
    
    p_train_step = pad_shard_unpad(p_train_step, static_argnums=(0, 2), static_return=True)
    p_eval_step = pad_shard_unpad(p_eval_step, static_argnums=(0, 2), static_return=True)
    
    # Wandb =======
    project_root = Path(BASE_DIR["project"])
    save_dir = configs.train_config.save_dir
    exp_name = configs.train_config.exp_name
    if use_wandb:
        run = wandb.init(project=exp_name.split("-")[0], dir=project_root, name=exp_name, config=configs.get_dict())
    else:
        run = None
    logger = Logger(run)
    
    bprint(f"Experiment: {exp_name}\nEnvironment: {configs.data_config.env_name}\nDevices: {n_devices}\nLogging to: {save_dir}\n", width=120)
    
    # Training loop =======
    state = train_prior(state, configs, p_train_step, p_eval_step, sample_batch_fn, dataloader,
                        logger=logger,
                        # eval_batch=eval_batch,
                        log_interval=log_interval,
                        save_interval=save_interval,
                        eval_freq=eval_freq,
                        **kwargs)
    
    # Save final model =======
    # save_state(state, save_dir / "checkpoint-final", total_steps)
    save_state(state, configs.train_config.save_dir / "checkpoint", total_steps)
    state = flax.jax_utils.unreplicate(state)
    configs.save(save_dir)
    pickle.dump({"params": state.params, **state.extra_variables}, open(save_dir / "model_params.pkl", "wb"))
    
    if log_interval > 0:
        logger.run.finish()


if __name__ == "__main__":
    import chex
    import os
    from src.models.tap_transformer import TAPWithHeads
    
    jax.config.update("jax_debug_nans", True)
    
    model_def = TAPWithHeads
    vae_path = Path(BASE_DIR["save"]) / "BERTAP_VAE-0625-1317"
    
    log_interval = 20
    save_interval = 2000
    eval_freq = 5
    pmap = True
    use_wandb = True
    test = False
    
    loader_size = 5 if test else 0
    batch_size = 256 if test else 512 * 4
    
    structure = {"emb_dim": 256,
                 "n_heads": 8,
                 "n_layers": 4,
                 "ff_dim": 256 * 4,
                 "causal": True,
                 "nsp_weight": 0.0,
                 "use_nsp": False}
    
    kwargs = {
        "model": {"modify_prob": 0.0, "mask_prob": 0.0, "random_prob": 0.0,
                  "n_special_tokens": 3, "vae_path": vae_path,
                  **structure},
        "dataset": {},
        "train": {"scheduler_name": "one_cycle", "learning_rate": 8e-3},
        "loader_size": loader_size
    }
    
    if pmap:
        main(model_def, vae_path, batch_size=batch_size, n_epochs=10,
             log_interval=log_interval, save_interval=save_interval, eval_freq=eval_freq, use_wandb=use_wandb, **kwargs)
        
    else:
        # jax.config.update("jax_platform_name", "cpu")
        # chex.set_n_cpu_devices(4)
        with chex.fake_pmap_and_jit():
            main(model_def, vae_path, batch_size=32, n_epochs=10,
                 log_interval=log_interval, save_interval=save_interval, eval_freq=eval_freq, use_wandb=use_wandb, **kwargs)
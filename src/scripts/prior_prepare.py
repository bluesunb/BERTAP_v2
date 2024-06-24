import pickle
import flax.jax_utils
import flax.traverse_util
import numpy as np
import jax, jax.numpy as jp
import flax, optax

from pathlib import Path

from src.models.vae import VQVAE
from src.models.transformer import BertWithHeads
from src.common.configs import TotalConfigs, DatasetConfig, ModelConfig, TrainConfig
from src.datasets import get_dataset, make_env, AntMLMDataLoader
from src.utils.context import make_rngs, make_state
from src.utils.logging_utils import get_now_str


def prepare_config_dataset(vae_path: Path,
                           batch_size: int,
                           n_epochs: int,
                           scheduler_name: str = "constant",
                           **kwargs) -> tuple[AntMLMDataLoader, TotalConfigs, flax.core.FrozenDict, TotalConfigs]:
    
    vae_config = TotalConfigs.load_from_txt(vae_path)       # load configs from text due to ease of modification
    vae_params = pickle.load((vae_path / "model_params.pkl").open("rb"))
    
    data_config = DatasetConfig(**vae_config.data_config.get_dict())
    data_config.update(**kwargs.get("dataset", {}))
    
    env = make_env(data_config.env_name)
    dataset = get_dataset(env, data_config.env_name)
    assert data_config.seq_len == data_config.min_valid_len, "seq_len != min_valid_len, There is a error in the saveing vae config"
    dataset = AntMLMDataLoader(dataset=dataset,
                               seq_len=data_config.seq_len,
                               min_valid_len=data_config.min_valid_len,
                               terminal_key=data_config.terminal_key,
                               goal_conditioned=data_config.goal_conditioned,
                               p_true_goal=data_config.p_true_goal,
                               p_sub_goal=data_config.p_sub_goal,
                               hierarchical_goal=data_config.hierarchical_goal)
        
    model_config = ModelConfig(**vae_config.model_config.get_dict())
    model_config.update(**dict(
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        ff_dim=768 * 4,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        emb_pdrop=0.1,
        ff_pdrop=0.1,
        
        causal=False,
        goal_conditional=True,
        hierarchical_goal=data_config.hierarchical_goal,
        vae_path=vae_path,

        n_special_tokens=3,
    ))
    model_config.update(**kwargs.get("model", {}))
    
    train_config = TrainConfig(exp_name=f"BERTAP_MLM-{get_now_str()}",
                               batch_size=batch_size,
                               n_epochs=n_epochs,
                               learning_rate=3e-4,
                               scheduler_name=scheduler_name,
                               grad_norm_clip=1.0,
                               betas=(0.9, 0.98),
                               weight_decay=0.01)
    train_config.update(**kwargs.get("train", {}))
    
    configs = TotalConfigs(data_config, model_config, train_config)
    return dataset, configs, vae_params, vae_config


def prepare_states(model_def: type[BertWithHeads],
                   configs: TotalConfigs,
                   scheduler: optax.Schedule,
                   model_kwargs: dict = None):
    
    model = model_def(configs.model_config, **(model_kwargs or {}))
    rng = jax.random.PRNGKey(configs.train_config.seed)
    
    @optax.inject_hyperparams
    def optimizer(schedule):
        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(configs.train_config.grad_norm_clip),
            optax.adamw(
                learning_rate=schedule,
                b1=configs.train_config.betas[0],
                b2=configs.train_config.betas[1],
                weight_decay=configs.train_config.weight_decay,
                mask=decay_mask_fn)
        )
        
    tx = optimizer(scheduler)
    input_ids = jp.empty((1, configs.model_config.reduced_len * 2 + 2), dtype='i4')
    conditions = jp.empty((1, configs.model_config.goal_dim + configs.model_config.observation_dim), dtype='f4')
    state = make_state(make_rngs(rng, ('dropout', ), contain_params=True), model, tx, input_ids, conditions)
    
    return state
    
    
def decay_mask_fn(params: optax.Params):
    flat_params = flax.traverse_util.flatten_dict(params)
    layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
    layer_norm_named_params = {
        layer[-2:]
        for layer_norm_name in layer_norm_candidates
        for layer in flat_params.keys()
        if any(map(lambda x: x.lower().startswith(layer_norm_name), layer))}
    
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params.keys()}
    return flax.traverse_util.unflatten_dict(flat_mask)
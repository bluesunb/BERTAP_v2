import jax
import jax.numpy as jp
import numpy as np
import optax

from src.datasets import get_dataset, make_env, AntDataLoader, Normalizer
from src.utils.context import make_rngs, make_state
from src.utils.logging_utils import get_now_str
from src.models.vae import VQVAE
from src.common.configs import TotalConfigs, DatasetConfig, ModelConfig, TrainConfig


def prepare_config_dataset(env_name: str,
                           seq_len: int,
                           latent_step: int,
                           batch_size: int,
                           n_epochs: int = 10,
                           **kwargs) -> tuple[AntDataLoader, TotalConfigs]:
    
    data_config = DatasetConfig(env_name=env_name, 
                                seq_len=seq_len, 
                                disable_goal=False, 
                                min_valid_len=seq_len)
    
    data_config.update(**kwargs.get("dataset", {}))
    
    env = make_env(data_config.env_name)
    dataset = get_dataset(env, data_config.env_name)
    dataset = AntDataLoader(dataset=dataset,
                            seq_len=data_config.seq_len,
                            min_valid_len=data_config.min_valid_len,
                            terminal_key=data_config.terminal_key,
                            goal_conditioned=data_config.goal_conditioned,
                            p_true_goal=data_config.p_true_goal,
                            p_sub_goal=data_config.p_sub_goal,
                            hierarchical_goal=data_config.hierarchical_goal)
    
    obs_dim = dataset.obs_dim
    act_dim = dataset.act_dim
    goal_dim = dataset.goal_dim
    transition_dim = goal_dim + obs_dim + act_dim + 1

    model_config = ModelConfig(
        transition_dim=transition_dim,
        observation_dim=obs_dim,
        action_dim=act_dim,
        goal_dim=goal_dim,
        hierarchical_goal=data_config.hierarchical_goal,

        causal=True,
        emb_dim=512,
        n_heads=8,
        n_layers=4,
        ff_dim=512 * 4,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        emb_pdrop=0.1,
        ff_pdrop=0.1,

        traj_emb_dim=512,
        n_traj_tokens=360,
        ma_update=False,

        seq_len=seq_len,
        latent_step=latent_step,
        goal_conditional=data_config.goal_conditioned,      # NOT USED WARNING
        multi_modal=False,                                  # NOT USED WARNING

        pos_weight=1.0,
        action_weight=5.0,
        masked_pred_weight=0.1,
        commit_weight=0.25,
        vq_weight=1.0
    )

    model_config.update(**kwargs.get("model", {}))

    train_config = TrainConfig(exp_name=f'BERTAP_VAE-{get_now_str()}',
                               batch_size=batch_size,
                               n_epochs=n_epochs,
                               learning_rate=8e-4,
                               grad_norm_clip=1.0,
                               betas=(0.9, 0.95),
                               weight_decay=1e-1)
    
    train_config.update(**kwargs.get("train", {}))

    configs = TotalConfigs(data_config, model_config, train_config)
    return dataset, configs


def prepare_states(model_def: type[VQVAE],
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
            optax.adamw(learning_rate=schedule,
                        b1=configs.train_config.betas[0],
                        b2=configs.train_config.betas[1],
                        weight_decay=configs.train_config.weight_decay)
        )
    
    tx = optimizer(scheduler)

    input_dim = configs.model_confi
    sample_input = jp.empty((1, configs.data_config.seq_len, input_dim), dtype=jp.float32)
    masks = jp.ones((1, configs.data_config.seq_len, 1), dtype=jp.float32)
    state = make_state(make_rngs(rng, ('vq', 'dropout'), contain_params=True),
                       model, tx, sample_input, masks, train=True)
    return state

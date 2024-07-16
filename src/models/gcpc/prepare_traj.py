import jax
import jax.numpy as jp
import numpy as np
import optax

from src.datasets import get_dataset, make_env, AntDataLoader, Normalizer
from src.utils.context import make_rngs, make_state
from src.utils.logging_utils import get_now_str
from src.models.vae import VQVAE
from src.common.configs import TotalConfigs, DatasetConfig, TrainConfig
from src.models.gcpc.configs import ModelConfig


def prepare_config_dataset(env_name: str,
                           seq_len: int,
                           batch_size: int,
                           window_size: int = 10,
                           future_size: int = 70,
                           n_epochs: int = 60,
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
                            goal_conditioned=True,
                            p_true_goal=data_config.p_true_goal,
                            p_sub_goal=data_config.p_sub_goal,
                            hierarchical_goal=data_config.hierarchical_goal)
    
    obs_dim = dataset.obs_dim
    act_dim = dataset.act_dim
    goal_dim = dataset.goal_dim

    model_config = ModelConfig(
        observation_dim=obs_dim,
        action_dim=act_dim,
        goal_dim=goal_dim,
        window_size=window_size,
        future_size=future_size,
        state_action=False,
        
        causal=False,
        emb_dim=256,
        n_heads=4,
        ff_dim=256 * 4,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        emb_pdrop=0.1,
        ff_pdrop=0.1,
        
        n_slots=4,
        n_enc_layers=2,
        n_dec_layers=1,
        use_goal=True,
        
        mask_prob=0.6
    )

    model_config.update(**kwargs.get("model", {}))

    train_config = TrainConfig(exp_name=f'GCPC_Traj',
                               batch_size=batch_size,
                               n_epochs=n_epochs,
                               learning_rate=1e-4,
                               grad_norm_clip=1.0,
                               betas=(0.9, 0.95),
                               weight_decay=1e-1)
    
    train_config.update(**kwargs.get("train", {}))
    train_config.update(exp_name=f'{train_config.exp_name}-{get_now_str()}')

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

    sample_batch = {
        "traj_seq": jp.empty((1, configs.model_config.seq_len, configs.model_config.features_dim)),
        "goal": jp.empty((1, 1, configs.model_config.goal_dim)),
        "mask": jp.ones((1, configs.model_config.seq_len)),
        "pad": jp.zeros((1, configs.model_config.seq_len))
    }

    state = make_state(make_rngs(rng, ('vq', 'dropout'), contain_params=True),
                       model, tx, train=True, **sample_batch)
    return state

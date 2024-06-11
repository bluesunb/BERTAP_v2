import jax
import jax.numpy as jp
import flax.linen as nn

from src.common.configs import ModelConfig
from src.tmp.dataloaders import Batch, init_batch
from src.datasets.dataset import Data
from src.common.codebook import VectorQuantizer, VQMovingAvg
from src.models.transformer import TransformerModule

from collections import namedtuple

EPSILON = 1e-12
Condition = namedtuple('Condition', ['observations', 'goals'])

default_kernel = nn.initializers.xavier_uniform()
init_normal = nn.initializers.normal(stddev=0.02)
init_const = nn.initializers.constant(0.0)


def split(arr: jp.ndarray, split_values: jax.typing.ArrayLike, accumulate: bool = True, axis: int = -1):
    if not accumulate:
        split_values = jp.cumsum(split_values)
    return jp.split(arr, split_values, axis=axis)[:len(split_values)]


class Encoder(nn.Module):
    config: ModelConfig
    
    @nn.compact
    # def __call__(self, traj_seq: jp.ndarray, mask: jp.ndarray = None, train: bool = True):
    def __call__(self, traj_batch: Batch, train: bool = True):
        
        traj_seq = jp.concatenate((traj_batch.goals, traj_batch.observations, traj_batch.actions, traj_batch.dones_float), axis=-1)
        x_enc = nn.Dense(self.config.emb_dim, kernel_init=init_normal, use_bias=False)(traj_seq)
        x_enc = nn.LayerNorm(epsilon=EPSILON)(x_enc)
        x_enc = TransformerModule(self.config, update_pos_emb=True)(x_enc, traj_batch.masks, train)
        x_enc = nn.max_pool(x_enc, window_shape=(self.config.latent_step, ), strides=(self.config.latent_step, ), padding='VALID')
        x_enc = nn.Dense(self.config.traj_emb_dim, kernel_init=init_normal)(x_enc)
        return x_enc
    
    
class Decoder(nn.Module):
    config: ModelConfig
    
    @nn.compact
    def __call__(self, latent_seq: jp.ndarray, conditions: Batch, train: bool = True):
        cond = jp.concatenate((conditions.goals, conditions.observations), axis=-1)
        cond = jp.expand_dims(cond, axis=1)
        latent_seq = jp.concatenate((cond.repeat(latent_seq.shape[1], axis=1), latent_seq), axis=-1)
        
        latent_seq = nn.Dense(self.config.emb_dim, kernel_init=init_normal, use_bias=False)(latent_seq)
        latent_seq = latent_seq.repeat(self.config.latent_step, axis=1)
        latent_seq = TransformerModule(self.config, update_pos_emb=True)(latent_seq, mask=None, train=train)
        latent_seq = nn.LayerNorm(epsilon=EPSILON)(latent_seq)
        
        traj_rec = nn.Dense(self.config.transition_dim + self.config.goal_input_dim, kernel_init=init_normal)(latent_seq)
        dims = jp.array([self.config.goal_input_dim, self.config.observation_dim, self.config.action_dim, 1])
        traj_rec = split(traj_rec, dims, accumulate=False, axis=-1)

        traj_rec = Batch(observations=traj_rec[1] + conditions.observations,
                         actions=traj_rec[2],
                         dones_float=traj_rec[3],
                         masks=jp.ones_like(traj_rec[3]),
                         goals=traj_rec[0] + conditions.goals)
        # traj_rec = traj_rec.at[..., :self.config.goal_input_dim + self.config.observation_dim].add(condition)
        return traj_rec


class VQVAE(nn.Module):
    config: ModelConfig
    training: bool = None

    def setup(self):
        self.encoder = Encoder(self.config, name='encoder')
        self.decoder = Decoder(self.config, name='decoder')

        if self.config.ma_update:
            self.vq = VQMovingAvg(self.make_rng('vq'), self.config.n_traj_tokens, self.config.commit_weight)
        else:
            self.vq = VectorQuantizer(self.config.n_traj_tokens, self.config.commit_weight)

        self.padding = jp.zeros((1, 1, self.config.transition_dim))
    
    def encode(self, traj_batch: Batch, train: bool = True):
        x_enc = self.encoder(traj_batch, train=train)
        return x_enc
    
    def decode(self, latent_seq: jp.ndarray, condition: jp.ndarray, train: bool = True):
        traj_preds = self.decoder(latent_seq, condition, train=train)
        return traj_preds
    
    def quantize(self, x: jp.ndarray, train: bool = True):
        quantized, info = self.vq(x, train=train)
        return quantized, info
    
    def __call__(self, traj_batch: Batch, train: bool = True):
        train = nn.merge_param('training', self.training, train)
        conditions = Condition(observations=traj_batch.observations[:, 0], goals=traj_batch.goals[:, 0])

        traj_enc = self.encode(traj_batch, train=train)
        quantized, vq_info = self.quantize(traj_enc, train=train)
        traj_rec = self.decode(quantized, conditions, train=train)
        return traj_rec, {"enc_vq": vq_info, "vq_loss": vq_info["vq_loss"]}
    
    
if __name__ == "__main__":
    import time
    import jax, optax
    from src.common.configs import TotalConfigs
    from src.utils.context import make_rngs
    from src.scripts.vae_prepare import prepare_config_dataset
    from pprint import pp

    env_name = "antmaze-large-play-v2"
    kwargs = {}
    dataset, configs = prepare_config_dataset(env_name, seq_len=64, latent_step=4, batch_size=1, **kwargs)

    rng = jax.random.PRNGKey(0)
    rng_x, rng_goal = jax.random.split(rng, 2)

    x = jax.random.uniform(rng_x, (1, 64, configs.model_config.transition_dim + configs.model_config.goal_input_dim), dtype=jp.float32)
    # goal = jax.random.uniform(rng_goal, (1, 64, configs.model_config.goal_dim), dtype=jp.float32)
    # batch = init_batch(batch_size=1,
    #                    seq_len=64,
    #                    obs_dim=configs.model_config.observation_dim,
    #                    act_dim=configs.model_config.action_dim,
    #                    goal_dim=configs.model_config.goal_dim)
    dims = jp.array([configs.model_config.goal_input_dim, configs.model_config.observation_dim, configs.model_config.action_dim, 1])
    x = split(x, dims, accumulate=False, axis=-1)
    batch = Batch(observations=x[1], actions=x[2], dones_float=x[3], masks=jp.ones_like(x[3]), goals=x[0])

    print(batch.goals.mean())

    model_keys = ('vq', 'dropout')
    model = VQVAE(configs.model_config)

    st = time.time()
    variables = model.init(make_rngs(rng, model_keys, contain_params=True), batch, train=True)
    print(f"init time: {time.time() - st:.4f}")

    def loss_fn(params, x, rng):
        (recon, vq_info), updates = model.apply(params, x, train=True,
                                                rngs=make_rngs(rng, model_keys), mutable=['vq_stats'])
        # loss = optax.l2_loss(x, recon).mean()
        # loss = jax.tree.map(lambda v1, v2: optax.l2_loss(v1, v2).mean(), x, recon)
        loss = jax.tree.map(lambda v1, v2: optax.l2_loss(v1, v2).mean(), x, recon)
        loss = sum(loss.values()) / len(loss)
        loss += vq_info['vq_loss']
        return loss, (vq_info, updates)
    
    grads, (vq_info, updates) = jax.grad(loss_fn, has_aux=True)(variables, batch, rng)
    pp(jax.tree.map(lambda x: jp.linalg.norm(x).item(), grads))

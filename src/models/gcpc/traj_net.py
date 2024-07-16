import numpy as np
import jax
import jax.numpy as jp
import flax.linen as nn
from src.models.gcpc.configs import ModelConfig
from src.models.transformer import TransformerEncoder

from transformers import FlaxBertForMaskedLM


init_normal = nn.initializers.normal(stddev=0.02)
init_const = nn.initializers.constant(0.0)

class PositionalEmbed(nn.Module):
    emb_dim: int
    max_len: int = None
    
    @nn.compact
    def __call__(self, x: jp.ndarray):
        max_len = self.max_len or x.shape[1]
        pe = jp.zeros((max_len, self.emb_dim), dtype=jp.float32)
        position = jp.arange(max_len)[..., jp.newaxis]
        div_term = jp.exp(jp.arange(0, self.emb_dim, 2) * -(jp.log(10000.0) / self.emb_dim))
        sin_pos = jp.sin(position * div_term)
        cos_pos = jp.cos(position * div_term)
        pe = jp.stack([sin_pos, cos_pos], axis=-2).transpose(0, 2, 1).reshape(max_len, self.emb_dim)
        return x + pe[jp.newaxis, :x.shape[1]]


class SlotEncoder(nn.Module):
    config: ModelConfig
    
    @nn.compact
    def __call__(self, traj_seq: jp.ndarray, mask: jp.ndarray, goal: jp.ndarray, train: bool = True):
        config = self.config
        bs = traj_seq.shape[0]
        
        traj_emb = nn.Dense(config.emb_dim, kernel_init=init_normal, use_bias=False)(traj_seq)
        goal_emb = nn.Dense(config.emb_dim, kernel_init=init_normal, use_bias=False)(goal)
        slot_emb = nn.Embed(config.n_slots, config.emb_dim, embedding_init=init_normal)(jp.arange(config.n_slots))
        slot_emb = jp.expand_dims(slot_emb, axis=0).repeat(bs, axis=0)
        
        x = jp.concatenate([slot_emb, goal_emb, traj_emb], axis=1)
        x = PositionalEmbed(config.emb_dim)(x)
        mask = jp.pad(mask, ((0, 0), (0, x.shape[1] - mask.shape[1])), constant_values=1)
        
        for _ in range(config.n_enc_layers):
            x = TransformerEncoder(config.emb_dim,
                                   config.n_heads, 
                                   config.ff_dim, 
                                   config.causal, 
                                   config.attn_pdrop, 
                                   config.resid_pdrop)(x, mask, train=train)
        
        return x
    

class SlotDecoder(nn.Module):
    config: ModelConfig
    
    @nn.compact
    def __call__(self, slot_seq: jp.ndarray, train: bool = True):
        config = self.config
        mask_seq = self.param('mask_traj', nn.initializers.zeros, (1, 1, config.emb_dim))
        mask_seq = jp.tile(mask_seq, (slot_seq.shape[0], config.seq_len, 1))
        
        z = jp.concatenate([slot_seq, mask_seq], axis=1)
        z = PositionalEmbed(config.emb_dim)(z)
        mask = jp.ones(z.shape[:2], dtype=jp.int32)
        
        for _ in range(config.n_dec_layers):
            z = TransformerEncoder(config.emb_dim,
                                   config.n_heads, 
                                   config.ff_dim, 
                                   config.causal, 
                                   config.attn_pdrop, 
                                   config.resid_pdrop)(z, mask=mask, train=train)
        
        recon = nn.Dense(self.config.observation_dim, kernel_init=init_normal)(z[:, -config.seq_len:])
        return recon
    
    
class TrajNet(nn.Module):
    config: ModelConfig
    training: bool = None
    
    def setup(self):
        self.encoder = SlotEncoder(self.config, name='slot_encoder')
        self.decoder = SlotDecoder(self.config, name='slot_decoder')
    
    def __call__(self, traj_seq: jp.ndarray, mask: jp.ndarray, goal: jp.ndarray, train: bool = None):
        train = nn.merge_param('training', self.training, train)
        slot_enc = self.encode(traj_seq, mask, goal, train=train)
        slot_dec = self.decode(slot_enc, train=train)
        return slot_dec
    
    def encode(self, traj_seq: jp.ndarray, mask: jp.ndarray, goal: jp.ndarray, train: bool = True):
        return self.encoder(traj_seq, mask, goal, train=train)
    
    def decode(self, slot_seq: jp.ndarray, train: bool = True):
        return self.decoder(slot_seq, train=train)

    
if __name__ == "__main__":
    config = ModelConfig(observation_dim=4,
                         action_dim=2,
                         goal_dim=2,
                         window_size=16,
                         future_size=24,
                         causal=False)
    
    rng = jax.random.PRNGKey(0)
    model = TrajNet(config)
    traj_seq = jax.random.uniform(rng, (32, config.window_size + config.future_size, config.observation_dim))
    mask = jp.concatenate([jp.zeros((32, config.window_size)), jp.ones((32, config.future_size))], axis=1, dtype=jp.int32)
    goal = traj_seq[:, -1:, :config.goal_dim]
    
    params = model.init({'params': rng, 'dropout': rng}, traj_seq, mask, goal)
    out = model.apply(params, traj_seq, mask, goal, rngs={'dropout': rng})
    print(out.shape)
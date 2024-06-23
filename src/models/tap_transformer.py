import jax
import jax.numpy as jp
import flax.linen as nn

from src.common.configs import ModelConfig
from src.models.transformer import TransformerEncoder

EPSILON = 1e-12
init_normal = nn.initializers.normal(stddev=0.02)


class TAPEmbedding(nn.Module):
    vocab_size: int
    max_seq_len: int
    emb_dim: int
    emb_pdrop: float

    @nn.compact
    def __call__(self, ids: jp.ndarray, condition: jp.ndarray, train: bool = True):
        seq_len = ids.shape[1]

        ids_emb = nn.Embed(self.vocab_size, self.emb_dim, name="ids_emb")(ids)
        pos_emb = self.param("pos_emb", nn.initializers.zeros, (1, self.max_seq_len, self.emb_dim), jp.float32)
        cond_emb = nn.Dense(self.emb_dim, name="cond_emb")(condition)

        ids_emb = jp.pad(ids_emb, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)

        emb = nn.LayerNorm(epsilon=EPSILON)(ids_emb + pos_emb[:, :seq_len + 1] + cond_emb)
        emb = nn.Dropout(self.emb_pdrop)(emb, deterministic=not train)
        return emb
    

class TAPWithHeads(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, 
                 input_ids: jp.ndarray, 
                 condition: jp.ndarray, 
                 mask: jp.ndarray = None, 
                 train: bool = True):
        
        if mask is None:
            mask = jp.ones((1, input_ids.shape[1], 1), dtype=input_ids.dtype)
        
        out = TAPEmbedding(self.config.vocab_size, self.config.seq_len, self.config.emb_dim, self.config.emb_pdrop)(input_ids[:, :-1], condition, train)
            
        for _ in range(self.config.n_layers):
            out = TransformerEncoder(self.config.emb_dim,
                                     self.config.n_heads,
                                     self.config.ff_dim,
                                     causal=True,
                                     attn_pdrop=self.config.attn_pdrop,
                                     resid_pdrop=self.config.resid_pdrop)(out, mask, train=train)
            
        logits = nn.Dense(self.config.vocab_size, kernel_init=init_normal, name="pred_head")(out)
        logits = logits.reshape((input_ids.shape[0], out.shape[1], -1))
        return logits
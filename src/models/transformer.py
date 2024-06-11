import jax, jax.numpy as jp
import flax.linen as nn
import flax, optax

from src.common.configs import ModelConfig
from src.models.attentions import MultiHeadAttention, FeedForward, positional_embedding, init_normal, init_const

EPSILON = 1e-12


class TransformerEncoder(nn.Module):
    emb_dim: int
    n_heads: int
    ff_dim: int
    causal: bool
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1

    @nn.compact
    def __call__(self, x: jp.ndarray, mask: jp.ndarray = None, train: bool = True):
        mask = jp.squeeze(mask, axis=-1)
        if mask.ndim - x.ndim < 1:
            attn_mask = nn.make_attention_mask(mask, mask, dtype=x.dtype)

        # out = nn.LayerNorm(epsilon=EPSILON)(x)
        attn_out = MultiHeadAttention(self.emb_dim,
                                      self.n_heads,
                                      self.attn_pdrop,
                                      self.causal)(x, x, x, attn_mask, train)

        attn_out = nn.Dropout(self.resid_pdrop)(attn_out, deterministic=not train)
        attn_out = nn.LayerNorm(epsilon=EPSILON)(x + attn_out)

        # ff_out = nn.LayerNorm(epsilon=EPSILON)(attn_out)
        ff_out = FeedForward(self.emb_dim, self.ff_dim)(attn_out)
        ff_out = nn.Dropout(self.resid_pdrop)(ff_out, deterministic=not train)
        enc = nn.LayerNorm(epsilon=EPSILON)(attn_out + ff_out)

        return enc


class TransformerModule(nn.Module):
    config: ModelConfig
    update_pos_emb: bool = False

    @nn.compact
    def __call__(self, x: jp.ndarray, mask: jp.ndarray = None, train: bool = True):
        if self.update_pos_emb:
            pos_ids = jp.arange(x.shape[1])[None, :]
            pos_emb = nn.Embed(self.config.seq_len, self.config.emb_dim, embedding_init=init_normal, name='pos_emb')(
                pos_ids)
        else:
            pos_emb = self.param("pos_emb", positional_embedding(self.config.seq_len, self.config.emb_dim))

        if mask is None:
            mask = jp.ones_like(x[..., 0:1])

        x_emb = nn.LayerNorm(epsilon=EPSILON)(x + pos_emb[:, :x.shape[1]])
        x_emb = nn.Dropout(self.config.emb_pdrop)(x_emb, deterministic=not train)

        for _ in range(self.config.n_layers):
            x_emb = TransformerEncoder(self.config.emb_dim,
                                       self.config.n_heads,
                                       self.config.ff_dim,
                                       self.config.causal,
                                       self.config.attn_pdrop,
                                       self.config.resid_pdrop)(x_emb, mask, train)

        return x_emb

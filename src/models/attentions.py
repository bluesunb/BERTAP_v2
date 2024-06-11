import jax
import jax.numpy as jp
import numpy as np
import flax, optax
import flax.linen as nn

from einops import rearrange
from typing import Callable

default_kernel = nn.initializers.xavier_uniform()
init_normal = nn.initializers.normal(stddev=0.02)
init_const = nn.initializers.constant(0.0)


def positional_embedding(max_len: int, emb_dim: int) -> Callable[[], jp.ndarray]:
    def init():
        pe = jp.zeros((max_len, emb_dim), dtype=jp.float32)
        position = jp.arange(max_len)[..., None]
        div_term = jp.exp(np.arange(0, emb_dim, 2) * (-jp.log(10000.0) / emb_dim))
        sin_pos = jp.sin(position * div_term)
        cos_pos = jp.cos(position * div_term)
        pe = jp.stack([sin_pos, cos_pos], axis=-2).transpose(0, 2, 1).reshape(max_len, emb_dim)
        return pe[None, ...]
    
    return init


class MultiHeadAttention(nn.Module):
    emb_dim: int
    n_heads: int
    attn_pdrop: float
    causal: bool = False
    
    def _split_heads(self, x: jp.ndarray):
        return rearrange(x, "b n (h d) -> b h n d", h=self.n_heads)
    
    def _merge_heads(self, x: jp.ndarray):
        return rearrange(x, "b h n d -> b n (h d)")
    
    @nn.compact
    def __call__(self, query: jp.ndarray, key: jp.ndarray, value: jp.ndarray, mask: jp.ndarray = None, train: bool = True):
        q = nn.Dense(self.emb_dim, kernel_init=default_kernel, name='q')(query)
        k = nn.Dense(self.emb_dim, kernel_init=default_kernel, name='k')(key)
        v = nn.Dense(self.emb_dim, kernel_init=default_kernel, name='v')(value)
        q, k, v = map(self._split_heads, (q, k, v))
        
        scale = self.emb_dim ** -0.5
        attn_weights = jp.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        
        # mask: (1, 1, seq_len, seq_len), 1 to mask out
        seq_len = query.shape[1]
        if mask is None:
            mask = jp.zeros((1, 1, seq_len, seq_len), dtype=jp.float32)
        
        if self.causal:
            causal_mask = nn.make_causal_mask(jp.ones((seq_len, )), extra_batch_dims=1)
            mask = nn.combine_masks(mask, causal_mask)
        
        attn_weights = mask * attn_weights + (1 - mask) * -1e9
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(self.attn_pdrop)(attn_weights, deterministic=not train)
        
        out = jp.einsum("b h i j, b h j d -> b h i d", attn_weights, v)
        out = self._merge_heads(out)
        out = nn.Dense(self.emb_dim, kernel_init=default_kernel, name='out')(out)
        return out
    

class FeedForward(nn.Module):
    emb_dim : int
    mid_emb_dim: int
    
    @nn.compact
    def __call__(self, x: jp.ndarray):
        x = nn.Dense(self.mid_emb_dim, kernel_init=default_kernel)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.emb_dim, kernel_init=init_normal)(x)
        return x
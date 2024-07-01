import jax
import jax.numpy as jp
import flax.linen as nn
import flax, optax

from src.common.configs import ModelConfig
from src.models.attentions import (
    MultiHeadAttention,
    FeedForward,
    positional_embedding,
    init_normal,
)

EPSILON = 1e-12


class TransformerEmbedding(nn.Module):
    max_seq_len: int
    emb_dim: int
    emb_pdrop: float
    update_pos_emb: bool = False

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        if self.update_pos_emb:
            pos_ids = jp.arange(x.shape[1])[None, :]
            pos_emb = nn.Embed(self.max_seq_len, self.emb_dim, embedding_init=init_normal, name="pos_emb")(pos_ids)
        else:
            pos_emb = self.param("pos_emb", positional_embedding(self.max_seq_len, self.emb_dim))

        x_emb = nn.LayerNorm(epsilon=EPSILON)(x + pos_emb[:, : x.shape[1]])
        x_emb = nn.Dropout(self.emb_pdrop)(x_emb, deterministic=not train)
        return x_emb


class BertEmbedding(nn.Module):
    vocab_size: int
    max_seq_len: int
    emb_dim: int
    emb_pdrop: float
    update_pos_emb: bool = True

    @nn.compact
    def __call__(self, ids: jp.ndarray, conditions: jp.ndarray, pos_ids: jp.ndarray, train: bool = True):
        ids_emb = nn.Embed(self.vocab_size, self.emb_dim, embedding_init=init_normal, name="ids_emb")(ids)
        pos_emb = nn.Embed(self.max_seq_len, self.emb_dim, embedding_init=init_normal, name="pos_emb")(pos_ids)
        cond_emb = nn.Dense(self.emb_dim, kernel_init=init_normal, name="cond_emb")(conditions)
        cond_emb = nn.tanh(cond_emb)

        emb = nn.LayerNorm(epsilon=EPSILON)(ids_emb + pos_emb + cond_emb)
        emb = nn.Dropout(self.emb_pdrop)(emb, deterministic=not train)
        return emb


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
        x_emb = TransformerEmbedding(self.config.max_seq_len,
                                     self.config.emb_dim,
                                     self.config.emb_pdrop,
                                     update_pos_emb=self.update_pos_emb)(x, train)

        if mask is None:
            mask = jp.ones_like(x[..., 0:1])

        for _ in range(self.config.n_layers):
            x_emb = TransformerEncoder(self.config.emb_dim,
                                       self.config.n_heads,
                                       self.config.ff_dim,
                                       self.config.causal,
                                       self.config.attn_pdrop,
                                       self.config.resid_pdrop)(x_emb, mask, train)

        return x_emb


class BertModule(nn.Module):
    config: ModelConfig
    update_pos_emb: bool = True
    add_pooling_layer: bool = False

    @nn.compact
    def __call__(self,
                 ids: jp.ndarray,
                 conditions: jp.ndarray,
                 pos_ids: jp.ndarray = None,
                 mask: jp.ndarray = None,
                 train: bool = True):

        if pos_ids is None:
            pos_ids = jp.arange(jp.atleast_2d(ids).shape[-1])
            pos_ids = jp.broadcast_to(pos_ids, ids.shape)

        if mask is None:
            mask = jp.ones_like(ids)[..., None]

        out = BertEmbedding(self.config.vocab_size,
                            self.config.max_seq_len,
                            self.config.emb_dim,
                            self.config.emb_pdrop,
                            self.update_pos_emb,
                            name="bert_emb")(ids, conditions, pos_ids, train=train)

        for _ in range(self.config.n_layers):
            out = TransformerEncoder(self.config.emb_dim,
                                     self.config.n_heads,
                                     self.config.ff_dim,
                                     causal=False,
                                     attn_pdrop=self.config.attn_pdrop,
                                     resid_pdrop=self.config.resid_pdrop)(out, mask, train=train)

        if self.add_pooling_layer:
            pooled_out = out[:, 0]
            pooled_out = nn.Dense(self.config.emb_dim, kernel_init=init_normal)(pooled_out)
            pooled_out = nn.tanh(pooled_out)
            return out, pooled_out

        return out, None


class PredictionHeadTransform(nn.Module):
    emb_dim: int

    @nn.compact
    def __call__(self, bert_out: jp.ndarray):
        bert_out = nn.Dense(self.emb_dim)(bert_out)
        bert_out = nn.gelu(bert_out)
        bert_out = nn.LayerNorm(epsilon=EPSILON)(bert_out)
        return bert_out


class VocabPredHead(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, bert_out: jp.ndarray, shared_embedding: jp.ndarray = None):
        x = PredictionHeadTransform(self.config.emb_dim)(bert_out)
        if shared_embedding is not None:
            x = jp.dot(x, shared_embedding.T)
        else:
            x = nn.Dense(self.config.vocab_size, use_bias=False)(x)
        
        bias = self.param("bias", nn.initializers.zeros, (self.config.vocab_size,))
        bias = jp.asarray(bias, dtype=jp.float32)
        return x + bias
    

class BinaryCLSHead(nn.Module):
    @nn.compact
    def __call__(self, pooling_out: jp.ndarray):
        return nn.Dense(2)(pooling_out)
    

class BertWithHeads(nn.Module):
    config: ModelConfig
    
    @nn.compact
    def __call__(self,
                 input_ids: jp.ndarray,
                 conditions: jp.ndarray,
                 pos_ids: jp.ndarray = None,
                 mask: jp.ndarray = None,
                 train: bool = True):
        
        bert_out, pooling_out = BertModule(self.config, add_pooling_layer=True, name="bert")(input_ids, conditions, pos_ids, mask, train)
        shared_emb = self.variables["params"]["bert"]["bert_emb"]["ids_emb"]["embedding"]
        
        vocab_logits = VocabPredHead(self.config, name="vocab_head")(bert_out, shared_emb)
        cls_logits = BinaryCLSHead(name="cls_head")(pooling_out)
        return vocab_logits, cls_logits
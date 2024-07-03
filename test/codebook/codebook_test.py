import jax
import jax.tree_util as jtr
import jax.numpy as jp
import flax.linen as nn
import optax

from boxprint import bprint

from src.models.codebook import (
    VectorQuantizer,
    VQMovingAvg,
    VQGumbel,
    VectorQuantizerEMA
)


class Net(nn.Module):
    model_def: nn.Module
    model_kwargs: dict = None
    
    def setup(self):
        model_kwargs = self.model_kwargs or {}
        self.vq = self.model_def(**model_kwargs)
        self.embed = nn.Embed(32, 128)
        
    def __call__(self, x: jp.ndarray, train: bool):
        x = self.embed(x)
        q, info = self.vq(x, train=train)
        return q, info


def loss_fn(params, model, x, **kwargs):
    (q, info), extra_variables = model.apply(params, x, train=True, mutable=['vq_stats'], **kwargs)
    enc = info.get('encodings')
    # return optax.softmax_cross_entropy_with_integer_labels(enc, x).mean()
    return info['vq_loss'] + q.mean()


if __name__ == "__main__":
    from pprint import pp
    from time import time

    rng = jax.random.PRNGKey(0)
    x = jp.array(jax.random.randint(rng, (1, 24, ), 0, 32))
    
    vq_rngs = jax.random.split(rng, 2)
    models = [
        Net(VectorQuantizer, {'codebook_size': 32}),
        Net(VQMovingAvg, {'rng': vq_rngs[0], 'codebook_size': 32}),
        Net(VQGumbel, {'codebook_size': 32}),
        Net(VectorQuantizerEMA, {'rng': vq_rngs[1], 'codebook_size': 32})
    ]
    
    rngs = jax.random.split(rng, 4)
    params = [model.init(rngs[i], jp.ones((1, 24), jp.int32), train=False) for i, model in enumerate(models)]
    
    for i, (param, model) in enumerate(zip(params, models)):
        st = time()
        context_rng = rngs[i]
        
        grad_fn = jax.value_and_grad(loss_fn)

        loss, grad = grad_fn(param, model, x, rngs={'gumbel': context_rng})
        print(f"{model.model_def.__name__} loss: {loss}")
        pp(jtr.tree_map(jp.linalg.norm, grad))
        print(f'{time() - st:=^20.4f}')
import jax
import jax.tree_util as jtr
import jax.numpy as jp
import flax.linen as nn
import optax

from typing import Any, Callable, Sequence

default_codebook_init = nn.initializers.variance_scaling(scale=1.0, mode='fan_in', distribution='uniform')


def uniform_init(lower: float, upper: float) -> Callable:
    def init(key, shape, dtype=jp.float32):
        return jp.random.uniform(key, shape, dtype, lower, upper)
    return init


def cdist(x: jp.ndarray, y: jp.ndarray, p: int = 2) -> jp.ndarray:
    return jp.linalg.norm(
        jp.expand_dims(x, axis=-2) - jp.expand_dims(y, axis=-3),
        ord=p, axis=-1
    )


def vector_indexization(inputs: jp.ndarray, codebook: jp.ndarray) -> jp.ndarray:
    distances = cdist(inputs, codebook)
    indices = jp.argmin(distances, axis=-1)
    return indices, distances


def vector_quantization(encodings: jp.ndarray, codebook: jp.ndarray) -> jp.ndarray:
    return jp.dot(encodings, codebook)


def vq_embedding(ids: jp.ndarray, codebook: jp.ndarray) -> jp.ndarray:
    return jp.take(codebook, ids, axis=0)


def calc_codebook_loss(x: jp.ndarray, quantized: jp.ndarray) -> jp.ndarray:
    return optax.l2_loss(jax.lax.stop_gradient(x), quantized).mean()


def calc_commitment_loss(x: jp.ndarray, quantized: jp.ndarray) -> jp.ndarray:
    return optax.l2_loss(x, jax.lax.stop_gradient(quantized)).mean()


def calc_entropy_loss(affinity: jp.ndarray, temperature: float = 1.0, eps: float = 1e-8) -> jp.ndarray:
    """
    Args:
        affinity: (bs, seq_len, n_tokens), output of the previous model, the logits
        temperature: temperature for the softmax
    """
    flat_affinity = affinity.reshape((-1, affinity.shape[-1]))
    flat_affinity /= temperature

    probs = jax.nn.softmax(flat_affinity, axis=-1)
    log_probs = jax.nn.log_softmax(flat_affinity + eps, axis=-1)

    target_probs = probs

    avg_probs = jp.mean(target_probs, axis=0)
    avg_entropy = -jp.sum(avg_probs * jp.log(avg_probs + eps))
    sample_entropy = -jp.mean(jp.sum(target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss


def get_perplexity(x: jp.ndarray, axis_name: str = "batch", eps: float = 1e-5):
    x = x.reshape(-1, x.shape[-1])
    x_probs = jp.mean(x, axis=0).astype(jp.float32)
    device_probs = jax.lax.pmean(x_probs, axis_name=axis_name)
    device_perplexity = jp.exp(-jp.sum(device_probs * jp.log(device_probs + eps)))
    perplexity = jp.exp(-jp.sum(x_probs * jp.log(x_probs + eps)))
    return perplexity, device_perplexity


class VectorQuantizer(nn.Module):
    codebook_size: int
    commit_loss_weight: float = 0.25
    entropy_loss_weight: float = 0.1
    entropy_temperature: float = 0.01

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        n_tokens = self.codebook_size
        codebook = self.param("codebook", default_codebook_init, (n_tokens, x.shape[-1])).astype(jp.float32)
        
        indices, distances = vector_indexization(x, codebook)
        encodings = jax.nn.one_hot(indices, n_tokens, dtype=jp.float32)
        quantized = vector_quantization(encodings, codebook)
        
        if train:
            commit_loss = calc_commitment_loss(x, quantized) * self.commit_loss_weight
            codebook_loss = calc_codebook_loss(x, quantized)
            entropy_loss = 0.0
            
            if self.entropy_loss_weight > 0.0:
                entropy_loss = calc_entropy_loss(-distances, self.entropy_temperature)
                entropy_loss *= self.entropy_loss_weight
                
            loss = codebook_loss + commit_loss + entropy_loss
            quantized = x + jax.lax.stop_gradient(quantized - x)
            vq_info = {
                "vq_loss": loss,
                "commit_loss": commit_loss,
                "codebook_loss": codebook_loss,
                "entropy_loss": entropy_loss
            }
            
        else:
            codebook_loss = optax.l2_loss(x, quantized).mean()
            vq_info = {"vq_loss": 0.0, "codebook_loss": codebook_loss}
            
        vq_info.update({"encodings": encodings, "indices": indices})
        return quantized, vq_info
    
    def get_codebook(self):
        return self.get_variable("param", "codebook").astype(jp.float32)
    
    
class VQMovingAvg(nn.Module):
    rng: jax.Array
    codebook_size: int
    commit_loss_weight: float = 0.25
    entropy_loss_weight: float = 0.1
    entropy_temperature: float = 0.01
    decay: float = 0.99
    
    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        n_tokens = self.codebook_size
        codebook = self.variable("vq_stats", "codebook",  default_codebook_init, self.rng, (n_tokens, x.shape[-1]))
        ema_weights = self.variable("vq_stats", "ema_weight", default_codebook_init, self.rng, (n_tokens, x.shape[-1]))
        counts = self.variable("vq_stats", "counts", nn.initializers.ones, self.rng, (n_tokens,))
        
        indices, distances = vector_indexization(x, codebook.value)
        encodings = jax.nn.one_hot(indices, n_tokens, dtype=jp.float32)
        
        if train:
            counts.value = self.decay * counts.value + (1.0 - self.decay) * jp.sum(encodings, axis=(0, 1))
            dw = jp.einsum('bln,bld->nd', encodings, x)     # Collect and sum the x according to the hotted indices => imagine the case encodings[:, 0] = 1
            ema_weights.value = self.decay * ema_weights.value + (1.0 - self.decay) * dw
            codebook.value = ema_weights.value / jp.expand_dims(counts.value, axis=-1)
            
        quantized = vector_quantization(encodings, codebook.value)
        quantized = x + jax.lax.stop_gradient(quantized - x)
        
        codebook_loss = optax.l2_loss(x, quantized).mean()
        vq_info = {"vq_loss": 0.0,
                   "codebook_loss": codebook_loss,
                   "commit_loss": 0.0,
                   "entropy_loss": 0.0,
                   "encodings": encodings,
                   "indices": indices}
        
        return quantized, vq_info
    
    def get_codebook(self):
        return self.get_variable("param", "codebook").astype(jp.float32)
    
    
class VQGumbel(nn.Module):
    codebook_size: int
    commit_loss_weight: float = 0.25
    entropy_loss_weight: float = 0.1
    entropy_temperature: float = 0.01
    
    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True, tau: float = 1.0):
        n_tokens = self.codebook_size
        codebook = self.param("codebook", default_codebook_init, (n_tokens, x.shape[-1])).astype(jp.float32)
        indices, distances = vector_indexization(x, codebook)
        
        if train:
            noise = jax.random.gumbel(self.make_rng("gumbel"), distances.shape, dtype=jp.float32)
            encodings = jax.nn.softmax((-distances + noise) / tau, axis=-1)
        else:
            encodings = jax.nn.one_hot(indices, n_tokens, dtype=jp.float32)
            
        quantized = vector_quantization(encodings, codebook)
        vq_info = {"vq_loss": 0.0,
                   "codebook_loss": 0.0,
                   "commit_loss": 0.0,
                   "entropy_loss": 0.0,
                   "encodings": encodings,
                   "indices": indices}
        
        return quantized, vq_info
    
    def get_codebook(self):
        return self.get_variable("param", "codebook").astype(jp.float32)
    
    
class ExpMovingAvg(nn.Module):
    rng: jax.Array
    shape: Sequence[int]
    decay: float = 0.99
    dtype: jp.dtype = jp.float32
    
    @nn.compact
    def __call__(self, value: jp.ndarray, train: bool = True):
        hidden = self.variable("vq_stats", "hidden", nn.initializers.zeros, self.rng, self.shape)
        average = self.variable("vq_stats", "average", nn.initializers.zeros, self.rng, self.shape)
        count = self.variable("vq_stats", "count", nn.initializers.zeros, self.rng, ())
        
        decay = jax.lax.convert_element_type(self.decay, value.dtype)
        one = jp.ones([], value.dtype)
        
        count_updated = count.value + 1
        hidden_updated = hidden.value * decay + (one - decay) * value
        average_updated = hidden_updated / (one - decay ** count_updated)
        
        if train:
            hidden.value = hidden_updated
            average.value = average_updated
            count.value = count_updated
            
        return average_updated
    
    
class VectorQuantizerEMA(nn.Module):
    rng: Any
    codebook_size: int
    commit_loss_weight: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5
    
    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        codebook = self.variable("vq_stats", "codebook", default_codebook_init, self.rng, (self.codebook_size, x.shape[-1]))
        rngs = jax.random.split(self.rng, 2)
        ema_cluster = ExpMovingAvg(rngs[0], (self.codebook_size,), decay=self.decay, dtype=jp.float32)
        ema = ExpMovingAvg(rngs[1], (self.codebook_size, x.shape[-1]), decay=self.decay, dtype=jp.float32)
        
        indices, distances = vector_indexization(x, codebook.value)
        encodings = jax.nn.one_hot(indices, self.codebook_size, dtype=jp.float32)
        
        if train:
            cluster_counts = encodings.sum(axis=(0, 1))
            cluster_counts = ema_cluster(cluster_counts, train=train)
            
            dw = jp.einsum('bln,bld->nd', encodings, x)
            dw = ema(dw, train=train)
            
            N = jp.sum(cluster_counts)
            cluster_counts = (cluster_counts + self.epsilon) / (N + self.codebook_size * self.epsilon) * N  # Laplace smoothing
            codebook.value = dw / jp.expand_dims(cluster_counts, axis=-1)
            
        quantized = vector_quantization(encodings, codebook.value)
        quantized = x + jax.lax.stop_gradient(quantized - x)
        codebook_loss = optax.l2_loss(x, quantized).mean()
        
        vq_info = {
            "vq_loss": 0.0,
            "codebook_loss": codebook_loss,
            "commit_loss": 0.0,
            "entropy_loss": 0.0,
            "encodings": encodings,
            "indices": indices
        }
        
        return quantized, vq_info


if __name__ == "__main__":
    from pprint import pp

    rng = jax.random.PRNGKey(0)


    class Net(nn.Module):
        tractable: bool

        @nn.compact
        def __call__(self, x, **kwargs):
            x = nn.Embed(32, 128)(x)
            q, info = VectorQuantizer(32)(x, **kwargs)
            encodings = info['encodings']
            return encodings


    model1 = Net(tractable=False)
    model2 = Net(tractable=True)
    params1 = model1.init(rng, jp.ones((1, 24), dtype=jp.int32))
    params2 = model2.init(rng, jp.ones((1, 24), dtype=jp.int32))


    def loss_fn(params, model, x):
        encodings = model.apply(params, x, train=True)
        return optax.softmax_cross_entropy_with_integer_labels(encodings, x).mean()


    x = jax.random.randint(rng, (1, 24), 0, 32)
    grad_fn = jax.value_and_grad(loss_fn)
    loss1, grad1 = grad_fn(params1, model1, x)
    loss2, grad2 = grad_fn(params2, model2, x)

    print(loss1, loss2)
    pp(grad1)
    pp(grad2)

    pp(jtr.map(jp.linalg.norm, grad1))
    pp(jtr.map(jp.linalg.norm, grad2))
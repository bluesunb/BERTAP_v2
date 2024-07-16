import pickle
import numpy as np
import jax
import jax.numpy as jp
import flax, optax
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import pad_shard_unpad



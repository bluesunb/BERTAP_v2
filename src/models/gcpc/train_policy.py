import pickle
import numpy as np
import jax
import jax.numpy as jp
import flax, optax
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import pad_shard_unpad

from src import BASE_DIR
from src.models.gcpc import TrajNet, PolicyNet
from src.models.gcpc.configs import ModelConfig
from src.datasets import AntDataLoader
from src.utils.train_state import TrainState
from src.utils.logging_utils import Logger
from src.utils.context import make_rngs

import wandb
from boxprint import bprint
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


def policy_train_step(state: TrainState, batch: Dict[str, jp.ndarray], rng: jp.ndarray, config: ModelConfig, pmap_axis=None):
    rng_names = ('dropout', )
    rngs = make_rngs(rng, rng_names)
    pad_mask = batch.pop('pad')[..., jp.newaxis]
    
    def loss_fn(params):
        actions = batch['actions']
        preds = state(**batch, train=True, params=params, rngs=rngs)
        loss = calc_action_loss(actions, preds, pad_mask)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name=pmap_axis)
    loss = jax.lax.pmean(loss, axis_name=pmap_axis)
    
    state = state.apply_gradients(grads=grads)
    return state, {'loss': loss}
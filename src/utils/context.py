import pickle
import os, shutil
import flax.jax_utils
import jax, jax.numpy as jp
import flax
from flax.training import checkpoints
from src.utils.train_state import TrainState

from pathlib import Path
from typing import Union, Sequence, Dict


def save_state(state: TrainState, path: Union[str, Path], step: int):
    if os.path.exists(path):
        shutil.rmtree(path)
    
    state = flax.jax_utils.unreplicate(state)
    state = jax.device_get(state)
    state = checkpoints.save_checkpoint(path, state, step)
    return state


def load_state(path: Union[str, Path]) -> TrainState:
    state_dict = checkpoints.restore_checkpoint(path)
    state = TrainState.create(**state_dict)
    return state


def load_params(path: Union[str, Path]) -> Dict[str, jp.ndarray]:
    params = pickle.load(open(os.path.join(path, "model_params.pkl"), 'rb'))
    return params


def make_rngs(rng: jp.ndarray,
              names: Union[Sequence[str], Dict[str, Sequence[int]]] = None,
              contain_params: bool = False):
    
    if names is None:
        return jax.random.split(rng)[1]
    
    elif isinstance(names, Sequence):
        rng, *rngs = jax.random.split(rng, len(names) + 1)
        rngs = {name: r for name, r in zip(names, rngs)}
        if contain_params:
            rngs["params"] = rng
        return rngs
    
    else:
        rngs = jax.random.split(rng, len(names))
        return {name: make_rngs(r, names[name], contain_params) for name, r in zip(names, rngs)}
    
    
def make_state(rngs, model, tx, *inputs, param_exclude=None, **kwargs) -> TrainState:
    variabales = model.init(rngs, *inputs, **kwargs)
    if param_exclude is not None:
        for key in param_exclude:
            if key in variabales["params"]:
                variabales["params"].pop(key)
                
    state = TrainState.create(model,
                              params=variabales.pop("params"),
                              tx=tx,
                              extra_variables=variabales)
    
    return state

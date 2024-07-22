import jax
import jax.numpy as jp
import jax.tree_util as jtr
from flax.training.common_utils import shard

from src.datasets import AntDataLoader, Normalizer
from src.models.gcpc.configs import ModelConfig

import numpy as np
from typing import Optional, Callable, Tuple, List, Dict, Literal


def gcpc_batch_sampler(
    loader: AntDataLoader,
    batch_size: int,
    normalize: bool,
    model_config: ModelConfig,
):
    window_size = model_config.window_size
    future_size = model_config.future_size
    
    assert loader.seq_len == window_size + future_size, \
        f"Loader seq_len {loader.seq_len} does not match window_size {window_size} + future_size {future_size}"
    
    normalizer = Normalizer(loader.dataset, keys=["goals", "observations", "actions"])
    
    def denormalize_fn(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        traj_seq = batch["traj_seq"]
        
        splits = [loader.obs_dim, loader.action_dim]
        splits = np.cumsum([0] + splits)
        splits = [(splits[i], splits[i+1]) for i in range(len(splits) - 1)]
        
        traj_seq = normalizer.denormalize_concat(traj_seq, keys=["observations", "actions"], splits=splits)
        batch["traj_seq"] = traj_seq
        batch = normalizer.denormalize(batch)
        return batch
    
    def sample_fn(pmap: bool = False, **sample_kwargs) -> Dict[str, np.ndarray]:
        batch = loader.sample(batch_size, **sample_kwargs)
        if normalize:
            batch = normalizer.normalize(batch)

        window_mask = np.random.rand(batch_size, window_size) > model_config.window_mask_rate
        future_mask = np.random.rand(batch_size, future_size) > model_config.future_mask_rate
        mask = np.concatenate([window_mask, future_mask], axis=1)
        batch = {"traj_seq": batch["observations"] if not model_config.state_action else \
                    np.concatenate([batch["observations"], batch["actions"]], axis=-1), 
                 "goal": batch["goals"][:, :1],
                 "mask": mask, 
                 "pad": 1- batch["dones_float"]}

        if pmap:
            return shard(batch)
        
        return batch
    
    return sample_fn, denormalize_fn

import jax
import jax.numpy as jp
import jax.tree_util as jtr
from flax.training.common_utils import shard

from src.datasets import AntDataLoader, Normalizer

import numpy as np
from typing import Optional, Callable, Tuple, List, Dict, Literal


def gcpc_batch_sampler(
    loader: AntDataLoader,
    batch_size: int,
    normalize: bool = True,
    only_state: bool = True,
    window_size: int = 16,
    future_size: int = 24,
    window_mask_rate: float = 0.2,
    future_mask_rate: float = 1.0,
):
    
    assert loader.seq_len == window_size + future_size, \
        f"Loader seq_len {loader.seq_len} does not match window_size {window_size} + future_size {future_size}"
    
    normalizer = Normalizer(loader.dataset, keys=["goals", "observations"])
    
    def sample_fn(pmap: bool = False, **sample_kwargs) -> Dict[str, np.ndarray]:
        batch = loader.sample(batch_size, **sample_kwargs)
        if normalize:
            batch = normalizer.normalize(batch)

        window_mask = np.random.rand(batch.shpae[0], window_size) < window_mask_rate
        future_mask = np.random.rand(batch.shpae[0], future_size) < future_mask_rate
        mask = np.concatenate([window_mask, future_mask], axis=1)[..., np.newaxis]
        batch = {"traj_seq": batch["observations"] if only_state else \
                    np.concatenate(batch["observations"], batch["actions"], axis=-1), 
                 "goal": batch["goals"][:, :1],
                 "mask": mask, 
                 "pad": batch["dones_float"]}

        if pmap:
            return shard(batch)
        
        return batch
    
    return sample_fn

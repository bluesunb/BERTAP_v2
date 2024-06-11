import numpy as np
from flax.training.common_utils import shard, shard_prng_key
from src.datasets import Normalizer, AntDataLoader
from typing import Optional, Callable, Tuple, List, Dict, Any


def vae_batch_sampler(loader: AntDataLoader, batch_size: int, normalize: bool = True) \
        -> Tuple[Callable[[], Dict[str, np.ndarray]], Tuple[Normalizer, List[Tuple[int, int]]]]:
    """
    Returns
         - sample_fn: Callable[[], Tuple[np.ndarray, np.ndarray]], a function that samples a batch of data
         - normalizer: Normalizer, it saves the mean and std of the data, and keys to normalize
         - splits: List[Tuple[int, int]], it saves the splits of the data to concatenate, which is used in normalization
    """
    tmp_data = loader.sample()
    keys = ('goals', 'observations', 'actions', loader.terminal_key)
    normalizer = Normalizer(loader.dataset, keys[:-1])
    
    dims = [tmp_data[key].shape[-1] if tmp_data[key].ndim > 1 else 1 for key in normalizer.mean.keys()]
    splits = np.cumsum([0] + dims)
    splits = [(splits[i - 1], splits[i]) for i in range(1, len(splits))]
    
    def sample_fn(pmap: Optional[bool] = False, **kwargs) -> Dict[str, np.ndarray]:
        batch = loader.sample(batch_size, **kwargs)
        if normalize:
            batch = normalizer.normalize(batch)
            
        batch = {"traj_seq": np.concatenate([batch[key] for key in keys], axis=-1), "masks": batch["masks"]}
        
        if pmap:
            batch = shard(batch)

        return batch
    
    return sample_fn, (normalizer, splits)

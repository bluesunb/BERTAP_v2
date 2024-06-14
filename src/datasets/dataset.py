import jax
import jax.tree_util as jtr
import numpy as np
from typing import Dict, List, Tuple

Data = Dict[str, np.ndarray]


def get_size(data: Dict):
    sizes = jtr.map(lambda arr: len(arr), data)
    return max(jax.tree.leaves(sizes))


class Dataset:
    def __init__(self, *args, **kwargs):
        self._dict: Data = dict(*args, **kwargs)
        self.size = get_size(self._dict)
        
    @classmethod
    def create(cls,
               observations: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               freeze: bool = True,
               **extra_fields) -> "Dataset":
        
        data = {"observations": observations,
                "actions": actions,
                "rewards": rewards,
                **extra_fields}
        
        data = jtr.map(lambda x: np.atleast_3d(x[None])[0], data)  # Make sure all fields has its feature dimension
        
        if freeze:
            jtr.map(lambda x: x.setflags(write=False), data)
            
        return cls(data)
    
    def sample(self, batch_size: int = None, index: np.ndarray = None, seed: int = None):
        assert not (batch_size is None and index is None), "Either batch_size or index must be provided"
        if index is None:
            generator = np.random.default_rng(seed)
            index = generator.integers(0, self.size, size=batch_size)
        return self.get_subset(index)
    
    def get_subset(self, index: np.ndarray) -> Data:
        return jtr.map(lambda x: x[index], self._dict)
    
    def keys(self) -> List[str]:
        return list(self._dict.keys())
    
    def items(self) -> List[Tuple[str, np.ndarray]]:
        return list(self._dict.items())
    
    def values(self) -> List[np.ndarray]:
        return list(self._dict.values())
    
    def copy(self) -> "Dataset":
        return Dataset(self._dict.copy())
    
    def __getitem__(self, key: str) -> np.ndarray:
        return self._dict[key]
    
    def __setitem__(self, key: str, value: np.ndarray):
        self._dict[key] = value
        
    def __iter__(self):
        for i in range(self.size):
            yield self.get_subset(np.array([i]))
            
    def __repr__(self):
        key_shapes = jtr.map(lambda x: x.shape, self._dict)
        return f"Dataset({', '.join(f'{k}: {v}' for k, v in key_shapes.items())})"
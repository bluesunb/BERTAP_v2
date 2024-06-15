import numpy as np
from src.datasets.dataset import Dataset
from typing import Sequence, Dict, NamedTuple


class Normalizer:
    def __init__(self, dataset: Dataset, keys: Sequence[int] = None, **kwargs):
        self.mean: Dict[str, np.ndarray] = {}
        self.std: Dict[str, np.ndarray] = {}
        
        keys = keys or dataset.keys()
        for key in keys:
            self.mean[key] = np.mean(dataset[key], axis=0)
            self.std[key] = np.std(dataset[key], axis=0)
            self.std[key][self.std[key] < 1e-6] = 1.0
            
        if kwargs.get('hierarchical_goal', False):
            self.mean['goals'] = np.tile(self.mean['goals'], (2,))
            self.std['goals'] = np.tile(self.std['goals'], (2,))
            
    def normalize(self, dataset: Dataset, keys: Sequence[str] = None) -> Dict[str, np.ndarray]:
        normalized_data = dataset.copy()
        keys = (keys or dataset.keys()) & self.keys
        for key in keys:
            normalized_data[key] = (normalized_data[key] - self.mean[key]) / self.std[key]
        return normalized_data
    
    def denormalize(self, dataset: Dataset, keys: Sequence[str] = None) -> Dict[str, np.ndarray]:
        denormalized_data = dataset.copy()
        keys = (keys or dataset.keys()) & self.keys
        for key in keys:
            denormalized_data[key] = denormalized_data[key] * self.std[key] + self.mean[key]
        return denormalized_data
    
    def normalize_tuple(self, batch: NamedTuple, keys: Sequence[str]) -> NamedTuple:
        normalized_data = {}
        batch_class = batch.__class__
        keys = (keys or batch._fields) & self.keys
        for key in keys:
            normalized_data[key] = (getattr(batch, key) - self.mean[key]) / self.std[key]
        return batch_class(**normalized_data)
    
    def denormalize_tuple(self, batch: NamedTuple, keys: Sequence[str]) -> NamedTuple:
        denormalized_data = {}
        batch_class = batch.__class__
        keys = (keys or batch._fields) & self.keys
        for key in keys:
            denormalized_data[key] = getattr(batch, key) * self.std[key] + self.mean[key]
        return batch_class(**denormalized_data)
        
    def normalize_concat(self, batch: np.ndarray, keys: Sequence[str], splits: Sequence[tuple[int]]) -> np.ndarray:
        normalized_batch = batch.copy()
        for key, (a, b) in zip(keys, splits):
            normalized_batch[..., a:b] = (batch[..., a:b] - self.mean[key]) / self.std[key]
        return normalized_batch
    
    def denormalize_concat(self, batch: np.ndarray, keys: Sequence[str], splits: Sequence[tuple[int]]) -> np.ndarray:
        denormalized_batch = batch.copy()
        for key, (a, b) in zip(keys, splits):
            denormalized_batch[..., a:b] = batch[..., a:b] * self.std[key] + self.mean[key]
        return denormalized_batch
    
    def normalize_sequence(self, sequence: Sequence[Dataset], keys: Sequence[str] = None) -> Sequence[Dataset]:
        return [self.normalize(data, keys) for data in sequence]
    
    def denormalize_sequence(self, sequence: Sequence[Dataset], keys: Sequence[str]) -> Sequence[Dataset]:
        return [self.denormalize(data, keys) for data in sequence]
    
    @property
    def keys(self) -> Sequence[str]:
        return list(self.mean.keys())


class AntNormalizer(Normalizer):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.mean['observations'][2:] = 0
        self.std['observations'][2:] = 1
        self.mean['next_observations'][2:] = 0
        self.std['next_observations'][2:] = 1

        if 'goals' in self.mean:
            self.mean['goals'][2:] = 0
            self.std['goals'][2:] = 1
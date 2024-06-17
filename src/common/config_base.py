import os, pickle
from dataclasses import dataclass, asdict, field
from flax.core import FrozenDict
from pathlib import Path

from enum import Enum
from typing import Any, Dict, Iterable

from src import BASE_DIR


class ConfigBase:
    def __str__(self) -> str:
        return self._repr()
    
    def __iter__(self) -> Iterable:
        return iter(self.get_dict())
    
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    
    def __len__(self) -> int:
        return len(self.get_dict())
    
    def get(self, key: str, default: Any) -> Any:
        return getattr(self, key, default)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f'Invalid key: {k}')
            setattr(self, k, v)
        return self

    def _repr(self) -> str:
        string = f'\nConfig: {self.__class__.__name__}\n'
        for key in sorted(self.__dict__.keys()):
            string += f'{key}: {self.__dict__[key]}\n'
        return string

    def get_dict(self) -> FrozenDict[str, Any] | Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop('project_root', None)
        d.pop('save_dir', None)
        return FrozenDict(d)

    def save(self, save_path_name: Path):
        # save_name = save_name.split('.')[0]
        # save_dir = Path(BASE_DIR['save'])
        # save_dir.mkdir(parents=True, exist_ok=True)

        # pkl_path = save_dir / f'{save_name}.pkl'
        # pickle.dump(self.get_dict(), pkl_path.open('wb'))
        # txt_path = save_dir / f'{save_name}.txt'
        # txt_path.write_text(str(self))

        # print(f'Saved config to {pkl_path}, {txt_path}')
        # return self
        save_path, save_name = save_path_name.parent, save_path_name.name
        save_path.mkdir(parents=True, exist_ok=True)
        save_name = save_name.split('.')[0]
        pkl_path = save_path / f'{save_name}.pkl'
        txt_path = save_path / f'{save_name}.txt'

        pickle.dump(self.get_dict(), pkl_path.open('wb'))
        txt_path.write_text(str(self))
        print(f'Saved config to {pkl_path}, {txt_path}')
        return self
    
    @classmethod
    def load(cls, path: str | Path) -> "ConfigBase":
        config = pickle.load(open(path, 'rb'))
        if isinstance(config, dict):
            return cls(**config)
        return cls(**config)
    

if __name__ == "__main__":
    config = ConfigBase()
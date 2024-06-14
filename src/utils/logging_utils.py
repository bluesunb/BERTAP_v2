import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import seaborn as sns
from flax.traverse_util import flatten_dict

import pickle
import wandb
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from numbers import Number

from src.utils.ant_viz import GoalReachingAnt, get_canvas_image
from typing import Dict, Optional, Sequence

ArrayLike = jax.typing.ArrayLike


class FakeRun:
    def __init__(self, path: Optional[Path] = None, _mem_limit: int = 10000):
        self.path = path
        self.log_dict = defaultdict(lambda: deque(maxlen=_mem_limit))
        self.steps = defaultdict(lambda: deque(maxlen=_mem_limit))
        self._mem_limit = _mem_limit

    def log(self, data: Dict[str, float | ArrayLike], step: int):
        for key, val in data.items():
            self.log_dict[key].append(val)
            self.steps[key].append(step)
            
            if len(self.log_dict[key]) == self._mem_limit:
                return self.flush()
            
    def save(self):
        if self.path is not None:
            logs = {'log_dict': self.log_dict, 'steps': self.steps}
            pickle.dump(self.log_dict, (self.path / 'log_dict.pkl').open('wb'))
            print(f'Saved logs to {self.path}')
            
    def clear(self):
        self.log_dict.clear()
        self.steps.clear()

    def flush(self):
        self.save()
        self.clear()
        
    def finish(self):
        self.save()
        
        
class Logger:
    def __init__(self, run: Optional[wandb.run] = None):
        self._is_fake = run is None
        self.run = run or FakeRun()
        
    def log_scalars(self, data: Dict[str, float], step: int):
        self.run.log(data, step)
        
    def log_arrays(self, data: Dict[str, ArrayLike], step: int):
        """
        Log array as images to the run.
        There are 3 types of image arrays:
            1. arr: Just array itself is an image
            2. map: Array can represent as a heatmap
            3. img: Array is an image with shape (H, W, 3)
        """
        for k, v in data.items():
            if k.endswith("_map") or k.endswith("_img"):
                self.array_to_img(k, v, step)
            elif k.endswith("_arr"):
                self.run.log({k: v if self._is_fake else wandb.Image(v)}, step=step)
            
    def log(self, data: Dict[str, float | ArrayLike], step: int, include_imgs: bool = False, prefix: str = ''):
        if prefix:
            data = {f'{prefix}/{k}': v for k, v in data.items()}
        
        scalar_data, array_data = split_data(data)
        self.log_scalars(scalar_data, step)
        if include_imgs:
            self.log_arrays(array_data, step)
        
    def array_to_img(self, tag: str, arr: jp.ndarray, step: int, reduce: int | Sequence[int] = 0):
        """Convert a JAX array into a wandb image and log it to the run."""
        fig, ax = plt.subplots()
        arr = reduce_array(arr, reduce)
        
        if tag.endswith("_map"):
            sns.heatmap(jax.device_get(arr), ax=ax)
        elif tag.endswith("_img"):
            ax.imshow(jax.device_get(arr))
            ax.axis('off')
            
        return self.plot_to_img(tag, fig, step)
            
    def plot_to_img(self, tag: str, plot: plt.Figure, step: int):
        """Convert a matplotlib plot into a wandb image and log it to the run."""
        canvas = agg.FigureCanvasAgg(plot)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        w, h = plot.canvas.get_width_height()
        img = img.reshape((h, w, -1))
        plt.close(plot)
        
        self.run.log({tag: img if self._is_fake else wandb.Image(img)}, step)
        return img
    
    
def get_now_str():
    return datetime.now().strftime("%m%d-%H%M")


def reduce_array(arr: jp.ndarray, reduce: int | Sequence[int] | str = 0):
    if arr.ndim > 3:
        if isinstance(reduce, int):
            reduce = (reduce, )
        if isinstance(reduce, Sequence):
            arr = arr[reduce]
        elif reduce == "mean":
            arr = arr.mean(axis=list(range(arr.ndim - 2)))
        elif reduce == "sum":
            arr = arr.sum(axis=list(range(arr.ndim - 2)))

    elif arr.ndim == 1:
        arr = arr[None, :]
        
    return arr


def split_data(data: Dict[str, ArrayLike]):
    data_types = {k: 0 if isinstance(v, Number) or v.squeeze().ndim == 0 else v.squeeze().ndim for k, v in data.items()}
    scalar_data = {k: float(v) for k, v in data.items() if data_types[k] == 0}
    array_data = {k: v for k, v in data.items() if data_types[k] > 0}
    return scalar_data, array_data


def compare_recons(logger: Logger,
                   env: GoalReachingAnt,
                   origs: np.ndarray,
                   recons: np.ndarray,
                   goal_dim: int,
                   global_step: int,
                   quantized: np.ndarray = None,
                   goal_conditioned: bool = False):

    assert (not goal_conditioned or goal_dim > 0), "Goal conditioned but goal_dim not provided"

    n_paths = recons.shape[0]
    figsize = np.round(np.array([2, 1]) * (n_paths + 2)).astype(int).tolist()
    fig, axes = plt.subplots(2, n_paths, figsize=figsize, tight_layout=True)
    canvas = agg.FigureCanvasAgg(fig)

    if n_paths == 1:
        axes = axes[:, None]

    for j in range(n_paths):
        new_xlim = np.array([np.inf, -np.inf])
        new_ylim = np.array([np.inf, -np.inf])

        if quantized is not None:
            title = '\n'.join(' '.join(str(x) for x in quantized[j, i:i+8].astype(int)) for i in range(0, len(quantized[j]), 8))
            axes[0, j].set_title(title, fontsize=8)

        for i in range(2):
            ax: plt.Axes = axes[i, j]
            paths = origs if i == 0 else recons
            obs_starts = goal_dim

            # goal = paths[j, 0, :4 if hierarchical_goal else 2]
            goal = paths[j, 0, goal_dim - 2:goal_dim] if goal_conditioned else None
            pos = paths[j, :, obs_starts: obs_starts + 2]
            mask = paths[j, :, -1] > 0.5

            c = np.linspace(0, 1, len(pos))
            ax.scatter(pos[~mask][:, 0], pos[~mask][:, 1], s=10, c=c[~mask], alpha=0.5, zorder=1)   # plot positions
            if goal_conditioned:
                ax.scatter(*goal, s=500, c='r', edgecolors='k', marker='*', alpha=0.8, zorder=3)    # plot goal

            tmp_xlim = np.array(ax.get_xlim())
            tmp_ylim = np.array(ax.get_ylim())
            new_xlim = np.array([min(new_xlim[0], tmp_xlim[0]), max(new_xlim[1], tmp_xlim[1])])
            new_ylim = np.array([min(new_ylim[0], tmp_ylim[0]), max(new_ylim[1], tmp_ylim[1])])

            env.draw(ax)
            x_lim = np.array(ax.get_xlim())
            y_lim = np.array(ax.get_ylim())
            u_x = (x_lim[1] - x_lim[0]) / 8
            u_y = (y_lim[1] - y_lim[0]) / 8

            new_xlim = (new_xlim - x_lim[0]) / u_x
            new_ylim = (new_ylim - y_lim[0]) / u_y
            new_xlim = np.array([np.floor(new_xlim[0]), np.ceil(new_xlim[1])]) * u_x + x_lim[0]
            new_ylim = np.array([np.floor(new_ylim[0]), np.ceil(new_ylim[1])]) * u_y + y_lim[0]

        for i in range(2):
            axes[i, j].set_xlim(new_xlim)
            axes[i, j].set_ylim(new_ylim)

    plt.tight_layout()
    img = get_canvas_image(canvas)
    plt.close(fig)
    logger.log_arrays({"Reconstruction_arr": img}, step=global_step)
    
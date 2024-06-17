import jax
import jax.tree_util as jtr
import numpy as np
from dataclasses import dataclass
from src.datasets.dataset import Dataset, Data
from typing import Sequence, NamedTuple


def broadcast_to(arr: np.ndarray, target_dim: int, axis: int = 0):
    shape = list(arr.shape)
    dim_diff = max(0, target_dim - arr.ndim)
    if dim_diff > 0:
        axis = axis if axis >= 0 else target_dim + axis
        shape = shape[:axis] + [1] * dim_diff + shape[axis:]
    return arr.reshape(shape)


def batched_randint(starts: np.ndarray, ends: np.ndarray, size: int | Sequence = 1):
    size = size if isinstance(size, Sequence) else (size, )
    x = np.random.rand(*(starts.shape + size))
    starts = broadcast_to(starts, x.ndim, axis=-1)
    ends = broadcast_to(ends, x.ndim, axis=-1)
    starts = np.minimum(starts, ends - 1)
    return np.floor(starts + x * (ends - starts)).astype(int)


# class Batch(NamedTuple):    # TODO: Structured batch input can't be used in model since it can't be used directly to calculate loss
#     observations: np.ndarray
#     actions: np.ndarray
#     rewards: np.ndarray
#     dones_float: np.ndarray
#     masks: np.ndarray
#     goals: np.ndarray = None


@dataclass
class TrajDataLoader:
    dataset: Dataset
    seq_len: int
    min_valid_len: int = 3
    terminal_key: str = "dones_float"
    goal_conditioned: bool = False
    
    def __post_init__(self):
        self.obs_dim = self.dataset["observations"].shape[-1]
        self.act_dim = self.dataset["actions"].shape[-1]
        self.goal_dim = self.dataset["goals"].shape[-1] if "goals" in self.dataset.keys() else 0
        
        self.terminal_ids = np.nonzero(self.dataset[self.terminal_key])[0]
        near_term_dist = np.searchsorted(self.terminal_ids, np.arange(self.dataset.size))
        near_term_dist = self.terminal_ids[near_term_dist] - np.arange(self.dataset.size)
        assert np.all(near_term_dist >= 0)
        
        self.valid_ids = np.where(near_term_dist >= self.min_valid_len)[0]
        self.size = len(self.valid_ids)

        # remove the goal
        if not self.goal_conditioned:
            self.dataset["goals"] = np.zeros_like(self.dataset["goals"])
        
    def _make_valid_indices(self, indices: np.ndarray):
        # Overflow check
        overflow = indices >= self.dataset.size
        
        # Episode inbounds check
        near_term_ids = np.searchsorted(self.terminal_ids, indices[:, 0, None])    # Find the terminal of given episode
        near_term_ids = self.terminal_ids[near_term_ids]
        after_term = indices > near_term_ids
        
        # Make mask to specify the invalid indices
        invalid_mask = np.logical_or(overflow, after_term)
        
        # Correct indices to prevent runtime-error
        indices[overflow] = self.dataset.size - 1
        indices = np.minimum(indices, near_term_ids)
        
        return indices, invalid_mask
    
    def _sample_ids(self, batch_size: int = 1, starts: np.ndarray = None):
        if starts is None:
            starts = np.random.choice(self.valid_ids, size=batch_size)[:, None]
        else:
            starts = starts[:, None]
            
        indices = np.arange(self.seq_len)[None, :] + starts     # (bs, seq_len)
        return indices
    
    def sample(self, batch_size: int = 1, starts: np.ndarray = None) -> Data:
        indices = self._sample_ids(batch_size, starts)
        indices, invalid_mask = self._make_valid_indices(indices)
        batch = self.dataset.get_subset(indices)
        batch_size = batch["observations"].shape[0]
        
        invalid_mask = np.atleast_3d(invalid_mask)  # (bs, seq_len, 1)
        batch = jax.tree_map(lambda arr: arr * (1 - invalid_mask), batch)
        
        batch["dones_float"] = invalid_mask
        batch["masks"] = 1 - invalid_mask
        if not self.goal_conditioned:
            batch["goals"] = np.zeros_like((batch_size, self.seq_len, self.goal_dim), dtype=np.float32)
            
        batch = jtr.tree_map(np.atleast_3d, batch)
        return batch
    
    
@dataclass
class AntDataLoader(TrajDataLoader):
    p_true_goal: float = 1.0
    p_sub_goal: float = 0.0
    hierarchical_goal: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        self.goal_dim = self.dataset["goals"].shape[-1]
        if self.hierarchical_goal:
            self.goal_dim *= 2
            assert self.p_true_goal + self.p_sub_goal == 1.0
            
    def _sample_sub_goal_ids(self, indices: np.ndarray):
        near_term_ids = np.searchsorted(self.terminal_ids, indices[:, 0])   # fine nearest terminals from start of the episode
        near_term_ids = self.terminal_ids[near_term_ids]
        sub_goal_ids = batched_randint(starts=indices[:, -1], ends=near_term_ids, size=1)   # goals within the episode
        return sub_goal_ids
    
    def sample(self, batch_size: int = 1, starts: np.ndarray = None) -> Data:
        """
        Sample a batch of trajectories from the dataset for AntMaze.
        
        Batch: {
            - observations: np.ndarray, (batch_size, seq_len, obs_dim),
            - actions: np.ndarray, (batch_size, seq_len, act_dim),
            - dones_float: np.ndarray, (batch_size, seq_len, 1),
            - masks: np.ndarray, (batch_size, seq_len, 1),
            - goals: np.ndarray, (batch_size, seq_len, goal_dim), => zeros if not goal_conditioned
        }
        """
        indices = self._sample_ids(batch_size, starts)
        indices, invalid_mask = self._make_valid_indices(indices)
        batch = self.dataset.get_subset(indices)
        invalid_mask = np.atleast_3d(invalid_mask)
        batch = jtr.tree_map(lambda arr: arr * (1 - invalid_mask), batch)
        batch_size = batch["observations"].shape[0]
        
        batch["dones_float"] = invalid_mask
        batch["masks"] = 1 - invalid_mask

        if self.goal_conditioned:
            sub_goal_ids = self._sample_sub_goal_ids(indices)
            goal_types = np.random.binomial(1, self.p_true_goal, size=(batch_size, 1))
            true_goals = self.dataset["goals"][sub_goal_ids]
            sub_goals = self.dataset["observations"][sub_goal_ids][..., :self.goal_dim // (1 + self.hierarchical_goal)]
            
            if self.hierarchical_goal:
                goals = np.concatenate([sub_goals, true_goals], axis=-1)
            else:
                goals = np.where(np.atleast_3d(goal_types), true_goals, sub_goals)
            
            batch["goals"] = goals.repeat(self.seq_len, axis=1)
        else:
            batch["goals"] = np.zeros((batch_size, self.seq_len, self.goal_dim), dtype=np.float32)
            
        batch = jtr.tree_map(np.atleast_3d, {
            "observations": batch["observations"],
            "actions": batch["actions"],
            "dones_float": batch["dones_float"],
            "masks": batch["masks"],
            "goals": batch["goals"]
        })
        return batch
    
    
@dataclass
class AntMLMDataLoader(AntDataLoader):
    def __post_init__(self):
        # assert self.goal_conditioned, "AntMLMDataLoader must be goal conditioned"
        self.hierarchical_goal_orig = self.hierarchical_goal
        self.goal_conditioned_orig = self.goal_conditioned
        self.hierarchical_goal = True
        self.goal_conditioned = True
        return super().__post_init__()
    
    def sample(self, batch_size: int = 1, starts: np.ndarray = None):
        batch1 = super().sample(batch_size, starts)
        batch2 = super().sample(batch_size, starts)
        
        goals1 = batch1["goals"][..., 0, -self.goal_dim:]   # true goals
        goals2 = batch2["goals"][..., 0, -self.goal_dim:]   # true goals
        nsp_labels = np.asarray(np.linalg.norm(goals1 - goals2, axis=-1) < 2.0, dtype=np.int32)
        
        if not self.goal_conditioned_orig:
            batch1["goals"] = np.zeros_like(batch1["goals"])
            batch2["goals"] = np.zeros_like(batch2["goals"])
            
        if not self.hierarchical_goal_orig:
            batch1["goals"] = batch1["goals"][..., -self.goal_dim // 2:]
            batch2["goals"] = batch2["goals"][..., -self.goal_dim // 2:]
            
        return batch1, batch2, nsp_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.datasets.d4rl_utils import make_env, get_dataset
    from src.utils.ant_viz import GoalReachingAnt
    
    env_name = 'antmaze-large-play-v2'
    env = make_env(env_name)
    render_env = GoalReachingAnt(env_name)
    dataset = get_dataset(env, env_name)
    ant_loader = AntDataLoader(dataset=dataset,
                               seq_len=64,
                               min_valid_len=64,
                               terminal_key='dones_float',
                               goal_conditioned=True,
                               p_true_goal=0.0,
                               p_sub_goal=0.1,
                               hierarchical_goal=False)

    def check(i):
        # idx = np.arange(ant_loader.terminal_ids[i], ant_loader.terminal_ids[i + 1]) + 1
        # batch = ant_loader.dataset.get_subset(idx)
        start = ant_loader.terminal_ids[i] + 1
        batch = ant_loader.sample(starts=np.array([start]))
        render_env.draw()
        c = np.linspace(0, 1, batch["observations"].shape[-2])
        pos = batch["observations"][..., :2]
        goal = batch["goals"][..., :2]
        
        plt.scatter(*pos.T, c=c, s=10, alpha=0.5, zorder=1)
        plt.scatter(*goal.T, c='r', s=100, edgecolors='k', marker='*', alpha=0.8, zorder=2)
        plt.show(block=True)
        
    check(20)
    check(20)
    check(30)
    check(30)
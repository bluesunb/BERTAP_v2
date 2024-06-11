import time
import gym
import numpy as np
from src.datasets.dataset import Dataset


def make_env(env_name: str):
    wrapped_env = gym.make(env_name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = env_name
    return env


def qlearning_dataset(env, dataset=None, terminate_on_end=False, disable_goal=False, **kwargs):
    """
    1. Get the dataset from the environment
    2. Extract the observations, actions, next_observations, rewards, terminals, and goals (if available)
    3. If the dataset has timeouts, extract the final timesteps
    4. Exclude the last timestep if terminate_on_end is False
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
        
    goals = dataset["infos/goal"] if not disable_goal and "infos/goal" in dataset else None
    use_timeouts = "timeouts" in dataset
    if use_timeouts:
        final_timesteps = dataset["timeouts"][:-1]
    else:
        max_ep_steps = env._max_episode_steps
        final_timesteps = np.zeros_like(dataset["rewards"])
        final_timesteps[max_ep_steps - 1::max_ep_steps] = True
        
    dataset = {
        "observations": dataset["observations"][:-1].astype(np.float32),
        "actions": dataset["actions"][:-1].astype(np.float32),
        "next_observations": dataset["observations"][1:].astype(np.float32),
        "rewards": dataset["rewards"][:-1].astype(np.float32),
        "terminals": dataset["terminals"][:-1].astype(bool),
    }
    
    if goals is not None:
        dataset["goals"] = goals[:-1].astype(np.float32)

    if not terminate_on_end:
        dataset = {k: v[~final_timesteps] for k, v in dataset.items()}

    return dataset


def get_dataset(env: gym.Env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                dataset=None,
                filter_terminals=False,
                disable_goal=False,
                obs_dtype=np.float32):
    
    if dataset is None:
        print("Loading dataset...")
        dataset = qlearning_dataset(env, terminate_on_end=False, disable_goal=disable_goal)
        
    if clip_to_eps:
        lim = 1 - eps
        dataset["rewards"] = np.clip(dataset["rewards"], -lim, lim)
    
    dataset["terminals"][-1] = 1
    if filter_terminals:
        non_term_ids = np.nonzero(~dataset["terminals"])[0]
        term_ids = np.nonzero(dataset["terminals"])[0]
        penalty_ids = term_ids - 1
        new_dataset = dict()
        
        for k, v in dataset.items():
            if k == "terminals":
                v[penalty_ids] = 1      # Set the terminal flag to 1 just before the terminal state
            new_dataset[k] = v[non_term_ids]
        
        dataset = new_dataset
        
    def antmaze_preprocess(dataset):
        print(f"==== [Fixing terminal for AntMaze] ====")
        dones_float = np.zeros_like(dataset["rewards"])
        dataset["terminals"][:] = 0
        
        ep_changed = np.linalg.norm(
            dataset["observations"][1:] - dataset["next_observations"][:-1], axis=-1
        ) > 1e-6
        
        ep_changed = ep_changed.astype(np.float32)
        dones_float[:-1] = ep_changed
        dones_float[-1] = 1
        return dones_float, dataset
    
    if "antmaze" in env_name:
        dones_float, dataset = antmaze_preprocess(dataset)
    else:
        dones_float = dataset["terminals"].copy()
        
    observations = dataset["observations"].astype(obs_dtype)
    next_observations = dataset["next_observations"].astype(obs_dtype)
    
    extra_kwargs = {}
    if "goals" in dataset:
        extra_kwargs["goals"] = dataset["goals"].astype(obs_dtype)
        
    return Dataset.create(
        observations=observations,
        actions=dataset["actions"].astype(np.float32),
        rewards=dataset["rewards"].astype(np.float32),
        masks=1.0 - dones_float.astype(np.float32),     # indicates whether the episode has ended
        dones_float=dones_float.astype(np.float32),     # ~mask
        next_observations=next_observations,
        **extra_kwargs
    )
    

class EpisodeMonitor(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._reset_state()
        self.total_timesteps = 0
        
    def _reset_state(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()
        
    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {'return': self.reward_sum,
                               'length': self.episode_length,
                               'time': time.time() - self.start_time}
            
            if hasattr(self, 'get_normalized_score'):
                ret = info['episode']['return']
                info['episode']['normalized_return'] = self.get_normalized_score(ret) * 100.0

        return observation, reward, done, info
    
    def reset(self):
        self._reset_stats()
        return self.env.reset()
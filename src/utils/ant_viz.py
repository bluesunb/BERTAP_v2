import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

import gym
import numpy as np
from typing import Callable, List
from functools import partial
from itertools import product, cycle

from d4rl.locomotion.ant import AntMazeEnv
from d4rl.pointmaze import MazeEnv


def get_canvas_image(canvas: FigureCanvasAgg):
    canvas.draw()
    out_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    out_img = out_img.reshape(canvas.get_width_height()[::-1] + (4,))
    return out_img[..., :3]


def valid_goal_sampler(env: AntMazeEnv, np_random: np.random.RandomState):
    valid_cells = []
    
    for i, j in product(range(len(env._maze_map)), range(len(env._maze_map[0]))):
        if env._maze_map[i, j] in (0, 'r', 'g'):
            valid_cells.append((i, j))
            
    cell = valid_cells[np.random.choice(len(valid_cells))]
    x, y = env._rowcol_to_xy(cell, add_random_noise=True)

    random_x, random_y = np.random.uniform(0, 0.5, size=(2,)) * 0.25 * env._maze_size_scaling
    x, y = max(x + random_x, 0), max(y + random_y, 0)
    return x, y


class GoalReachingAnt(gym.Wrapper):
    def __init__(self, env_name: str):
        self.env: AntMazeEnv = gym.make(env_name)
        if hasattr(self.env.env.env, '_wrapped_env'):
            self.env.env.env._wrapped_env.goal_sampler = partial(valid_goal_sampler, self.env.env.env._wrapped_env)
        else:
            self.env.env.env.goal_sampler = partial(valid_goal_sampler, self.env.env.env)
        self.observation_space = gym.spaces.Dict({'observation': self.env.observation_space,
                                                  'goal': self.observation_space})
        self.action_space = self.env.action_space

    def get_obs(self, obs: np.ndarray):
        target_goal = obs.copy()
        target_goal[:2] = self.target_goal
        return {'observation': obs, 'goal': target_goal}

    def reset(self):
        obs = self.env.reset()
        return self.get_obs(obs)

    def get_starting_boundary(self):
        env: AntMazeEnv = self.env.env.env
        torso_x, torso_y = env._init_torso_x, env._init_torso_y
        scale = env._maze_size_scaling

        bottom_left = (0 + 0.5 * scale - torso_x, 0 + 0.5 * scale - torso_y)
        top_right = ((len(env._maze_map[0]) - 1.5) * scale - torso_x, (len(env._maze_map) - 1.5) * scale - torso_y)
        return bottom_left, top_right

    def get_grid(self, n: int = 20):
        bottom_left, top_right = self.get_starting_boundary()
        x = np.linspace(bottom_left[0] + 0.04 * (top_right[0] - bottom_left[0]),
                        top_right[0] - 0.04 * (top_right[0] - bottom_left[0]), n)
        y = np.linspace(bottom_left[1] + 0.04 * (top_right[1] - bottom_left[1]),
                        top_right[1] - 0.04 * (top_right[1] - bottom_left[1]), n)

        X, Y = np.meshgrid(x, y)
        states = np.array([X.flatten(), Y.flatten()]).T
        return states

    def four_goals(self):
        env: AntMazeEnv = self.env.env.env
        valid_cells = []

        # for i, j in product(range(len(env._maze_map), range(len(env._maze_map[0])))):
        #     if env._maze_map[i, j] in [0, 'r', 'g']:
        #         valid_cells.append(env._rowcol_to_xy((i, j), add_random_noise=False))

        map = np.array(env._maze_map)
        for i, j in np.argwhere((map == '0') | (map == 'r') | (map == 'g')):
            valid_cells.append(env._rowcol_to_xy((i, j), add_random_noise=False))

        goals = [max(valid_cells, key=lambda x: -x[0] - x[1]),  # top left
                 max(valid_cells, key=lambda x: x[0] - x[1]),  # bottom left
                 max(valid_cells, key=lambda x: x[0] + x[1]),  # bottom right
                 max(valid_cells, key=lambda x: -x[0] + x[1])]  # top right

        return goals

    def draw(self, ax=None):
        if not ax:
            ax = plt.gca()

        env: AntMazeEnv = self.env.env.env
        torso_x, torso_y = env._init_torso_x, env._init_torso_y
        scale = env._maze_size_scaling

        map = np.array(env._maze_map)
        for i, j in np.argwhere(map == '1'):
            rect = patches.Rectangle(xy=(j * scale - torso_x - 0.5 * scale, i * scale - torso_y - 0.5 * scale),
                                     width=scale, height=scale,
                                     linewidth=1, edgecolor='none', facecolor='grey', alpha=1.0)
            ax.add_patch(rect)

        ax.set_xlim(0 + 0.1 * scale - torso_x, len(env._maze_map[0]) * scale - 1.1 * scale - torso_x)
        ax.set_ylim(0 + 0.1 * scale - torso_y, len(env._maze_map) * scale - 1.1 * scale - torso_y)
        ax.axis('off')
        
        
class GoalReachingMaze(GoalReachingAnt):
    def get_starting_boundary(self):
        env: MazeEnv = self.env.env.env
        init_x, init_y = env.init_qpos[:2]
        scale = 1.0
        
        bottom_left = (0 + 0.5 * scale - init_x, 0 + 0.5 * scale - init_y)
        top_right = ((env.maze_arr.shape[1] - 1.5) * scale - init_x, (env.maze_arr.shape[0] - 1.5) * scale - init_y)
        return bottom_left, top_right
    
    def draw(self, ax=None):
        if not ax:
            ax = plt.gca()
        
        env: MazeEnv = self.env.env.env
        init_x, init_y = env.init_qpos[:2]
        scale = 1.0
        
        map = np.array(env.maze_arr)
        for i, j in np.argwhere(map == 10):
            rect = patches.Rectangle(xy=(j * scale - init_x - 0.5 * scale, i * scale - init_y - 0.5 * scale),
                                     width=scale, height=scale,
                                     linewidth=1, edgecolor='none', facecolor='grey', alpha=1.0)
            ax.add_patch(rect)
            
        ax.set_xlim(0 + 0.1 * scale - init_x, env.maze_arr.shape[1] * scale - 1.1 * scale - init_x)
        ax.set_ylim(0 + 0.1 * scale - init_y, env.maze_arr.shape[0] * scale - 1.1 * scale - init_y)
        ax.axis('off')


def plot_trajectories(env: GoalReachingAnt,
                      trajectories: np.ndarray,
                      fig: plt.Figure,
                      ax: plt.Axes,
                      color_list: List[str] = None):
    if color_list is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_list = cycle(color_cycle)

    for i, (color, trajectory) in enumerate(zip(color_list, trajectories)):
        obs = np.array(trajectory['observations'])
        goal = np.array(trajectory['goals'][0])
        all_x, all_y = obs[:, 0], obs[:, 1]
        ax.scatter(all_x, all_y, s=10, c=color, alpha=0.5, zorder=0)
        ax.scatter(goal[0], goal[1], s=500, c=color, edgecolors='black', marker='*', alpha=0.9, zorder=1, label=str(i))

    env.draw(ax)


def trajectory_img(env: GoalReachingAnt, trajectories: np.ndarray, **kwargs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvasAgg(fig)

    plot_trajectories(env, trajectories, fig, plt.gca(), **kwargs)
    plt.tight_layout()
    img = get_canvas_image(canvas)
    plt.close(fig)
    return img


def gc_value_img(env: GoalReachingAnt, dataset, value_fn: Callable[[np.ndarray], np.ndarray]):
    base_observation = dataset['observations'][0]
    p1, p2, p3, p4 = env.four_goals()
    p3 = (32.75, 24.75)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvasAgg(fig)

    points = [p1, p2, p3, p4]
    for i, point in enumerate(points):
        point = np.array(point)
        ax = fig.add_subplot(2, 2, i + 1)

        goal_obs = base_observation.copy()
        goal_obs[:2] = point

        ax.set_title(f'Goal: {point[0]:.3f}, {point[1]:.3f}')
        ax.scatter(point[0], point[1], s=50, c='r', marker='*')

    img = get_canvas_image(canvas)
    plt.close(fig)
    return img
import jax
import jax.tree_util as jtr
import numpy as np
from src.datasets.dataloaders import AntDataLoader, AntMLMDataLoader


def sample_and_visualizing_test():
    import matplotlib.pyplot as plt
    from src.datasets.d4rl_utils import make_env, get_dataset
    from src.utils.ant_viz import GoalReachingAnt
    
    env_name = 'antmaze-large-play-v2'
    env = make_env(env_name)
    render_env = GoalReachingAnt(env_name)
    dataset = get_dataset(env_name)
    ant_loader = AntDataLoader(dataset=dataset,
                               seq_len=64,
                               min_valid_len=64,
                               terminal_key='dones_float',
                               goal_conditioned=True,
                               p_true_goal=0.0,
                               p_sub_goal=0.1,
                               hierarchical_goal=False)
    
    def check(i):
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
    
    
def ant_mlm_dataloader_validity_test():
    import time
    from src.datasets.d4rl_utils import make_env, get_dataset
    env_name = 'antmaze-large-play-v2'
    env = make_env(env_name)
    dataset = get_dataset(env, env_name)
    ant_loader = AntMLMDataLoader(dataset=dataset,
                                  seq_len=128,
                                  min_valid_len=128,
                                  terminal_key='dones_float',
                                  goal_conditioned=True,
                                  p_true_goal=0.0,
                                  p_sub_goal=1.0,
                                  hierarchical_goal=False)
    
    num_nsp = []
    st = time.time()
    for i in range(300):
        batch1, batch2, nsp_labels = ant_loader.sample(128)
        num_nsp.append(nsp_labels.mean())
    
    tot_time = time.time() - st
    print(f"Time taken: {tot_time:.2f}s (Mean: {tot_time / 100:.2f}s)")
    print(f"Mean number of NSP: {np.mean(num_nsp)}")
    
    
def dataloader_speed_test():
    import time
    from src.datasets.d4rl_utils import make_env, get_dataset
    env_name = 'antmaze-large-play-v2'
    env = make_env(env_name)
    dataset = get_dataset(env, env_name)
    
    ant_loader = AntDataLoader(dataset=dataset,
                               seq_len=128,
                               min_valid_len=128,
                               terminal_key='dones_float',
                               goal_conditioned=True,
                               p_true_goal=0.0,
                               p_sub_goal=1.0,
                               hierarchical_goal=False)
    
    ant_mlm_loader = AntMLMDataLoader(dataset=dataset,
                                      seq_len=128,
                                      min_valid_len=128,
                                      terminal_key='dones_float',
                                      goal_conditioned=True,
                                      p_true_goal=0.0,
                                      p_sub_goal=1.0,
                                      hierarchical_goal=False)
    
    batch_size = 512
    
    st = time.time()
    for i in range(200):
        ant_loader.sample(512)
        ant_loader.sample(512)
    print(f"AntDataLoader: {time.time() - st:.2f}s (Mean: {(time.time() - st) / 100:.2f}s)")
    
    st = time.time()
    for i in range(200):
        ant_mlm_loader.sample(512)
    print(f"AntMLMDataLoader: {time.time() - st:.2f}s (Mean: {(time.time() - st) / 100:.2f}s)")
    
    
if __name__ == "__main__":
    ant_mlm_dataloader_validity_test()
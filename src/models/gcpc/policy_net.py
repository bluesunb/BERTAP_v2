import jax
import jax.numpy as jp
import flax.linen as nn

from src.models.gcpc.configs import ModelConfig
from src.models.gcpc.traj_net import TrajNet


init_normal = nn.initializers.normal(stddev=0.02)
init_const = nn.initializers.constant(0.0)

class MLP(nn.Module):
    emb_dim: int
    out_dim: int
    ff_pdrop: float = 0.1
    
    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        x = nn.Dense(self.emb_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.ff_pdrop)(x, deterministic=not train)
        x = nn.Dense(self.out_dim)(x)
        return x
        

class PolicyNet(nn.Module):
    encoder: TrajNet
    encoder_param: dict
    config: ModelConfig
    
    @nn.compact
    def __call__(self, traj_seq: jp.ndarray, mask: jp.ndarray, goal: jp.ndarray, train: bool = True):
        config = self.config
        latent_seq = self.encoder.apply(self.encoder_param, traj_seq, mask, goal, train=False, method=self.encoder.encode)
        latent_seq = latent_seq.reshape(latent_seq.shape[0], 1, -1)
        obs_goal = jp.concat([goal, traj_seq[:, -1:]], axis=-1)
        
        obs_goal = nn.Dense(config.emb_dim, kernel_init=init_normal, use_bias=False)(obs_goal)
        latent_seq = nn.Dense(config.emb_dim, kernel_init=init_normal, use_bias=False)(latent_seq)
        x = nn.relu(jp.concatenate([obs_goal, latent_seq], axis=-1))
        a = MLP(config.emb_dim, config.action_dim, config.ff_pdrop)(x, train=train)
        return a


if __name__ == "__main__":
    config = ModelConfig(observation_dim=4,
                         action_dim=2,
                         goal_dim=2,
                         window_size=16,
                         future_size=24,
                         emb_dim=1024,
                         ff_pdrop=0.1)
    
    traj_config = ModelConfig(observation_dim=config.observation_dim,
                              action_dim=config.action_dim,
                              goal_dim=config.goal_dim,
                              window_size=config.window_size,
                              future_size=config.future_size,
                              emb_dim=256)
    
    rng = jax.random.PRNGKey(0)
    traj_seq = jax.random.uniform(rng, (32, config.window_size + config.future_size, config.observation_dim))
    mask = jp.concatenate([jp.zeros((32, config.window_size)), jp.ones((32, config.future_size))], axis=1, dtype=jp.int32)
    goal = traj_seq[:, -1:, :config.goal_dim]
    
    traj_net = TrajNet(traj_config)
    traj_param = traj_net.init(rng, traj_seq, mask, goal, train=False)
    
    rng, _ = jax.random.split(rng)
    model = PolicyNet(traj_net, traj_param, config)
    param = model.init(rng, traj_seq, mask, goal, train=False)
    out = model.apply(param, traj_seq, mask, goal, train=True, rngs={'dropout': rng})
    print(out.shape)
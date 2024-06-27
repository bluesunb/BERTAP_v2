import pickle
import numpy as np
import jax, jax.numpy as jp
from pathlib import Path

from src.common.configs import TotalConfigs, ModelConfig
from src.models.vae import VQVAE
from src.scripts.vae_prepare import prepare_config_dataset
from src.scripts.batch_samplers import vae_batch_sampler
from src.utils.logging_utils import compare_recons
from src.utils.ant_viz import GoalReachingAnt


#%%
render_env = GoalReachingAnt('antmaze-large-play-v2')

vae_path = Path.home() / "PycharmProjects/tmp/BERTAP_v2/save/BERTAP_VAE-0625-1317"
loader, configs = prepare_config_dataset('antmaze-large-play-v2', seq_len=96, latent_step=4, batch_size=4, n_epochs=10)

configs = TotalConfigs.load_from_txt(vae_path)
params = pickle.load((vae_path / "model_params.pkl").open("rb"))

batch_size = 4
denorm_keys = ('goals', 'observations', 'actions')
sample_batch_fn, (normalizer, splits) = vae_batch_sampler(loader, batch_size, normalize=True, hierarchical_goal=configs.data_config.hierarchical_goal)

eval_starts = np.arange(4) * loader.seq_len + 10 * 1000
eval_batch = sample_batch_fn(starts=eval_starts, pmap=False)

eval_batch_long = jax.tree.map(
    lambda x: jp.stack([jp.concatenate(x[:2], axis=0), jp.concatenate(x[2:], axis=0)], axis=0),
    eval_batch
)

#%%
model = VQVAE(configs.model_config)
recon, info = model.apply(params, **eval_batch, train=False)

compare_recons(None, render_env,
                normalizer.denormalize_concat(jax.device_get(eval_batch['traj_seq']), keys=denorm_keys, splits=splits),
                normalizer.denormalize_concat(jax.device_get(recon), keys=denorm_keys, splits=splits),
                goal_dim=configs.model_config.goal_dim,
                global_step=0,
                quantized=info['enc_vq']['indices'],
                goal_conditioned=configs.data_config.goal_conditioned,
                visualize=True)
#%%
recon_long, info = model.apply(params, **eval_batch_long, train=False)

compare_recons(None, render_env,
                    normalizer.denormalize_concat(jax.device_get(eval_batch_long['traj_seq']), keys=denorm_keys, splits=splits),
                    normalizer.denormalize_concat(jax.device_get(recon_long), keys=denorm_keys, splits=splits),
                    goal_dim=configs.model_config.goal_dim,
                    global_step=0,
                    quantized=info['enc_vq']['indices'],
                    goal_conditioned=configs.data_config.goal_conditioned,
                    visualize=True)

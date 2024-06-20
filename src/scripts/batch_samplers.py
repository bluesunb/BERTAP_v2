import jax
import jax.numpy as jp
import jax.tree_util as jtr
import flax.struct
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import pad_shard_unpad

from src.models.vae import VQVAE
from src.datasets import Normalizer, AntDataLoader, AntMLMDataLoader
from src.common.configs import ModelConfig

import numpy as np
from typing import Optional, Callable, Tuple, List, Dict, Any


def vae_batch_sampler(loader: AntDataLoader, batch_size: int, normalize: bool = True, **kwargs) \
        -> Tuple[Callable[[], Dict[str, np.ndarray]], Tuple[Normalizer, List[Tuple[int, int]]]]:
    """
    Returns
         - sample_fn: Callable[[], Tuple[np.ndarray, np.ndarray]], a function that samples a batch of data
         - normalizer: Normalizer, it saves the mean and std of the data, and keys to normalize
         - splits: List[Tuple[int, int]], it saves the splits of the data to concatenate, which is used in normalization
    """
    tmp_data = loader.sample()
    keys = ('goals', 'observations', 'actions', loader.terminal_key)
    normalizer = Normalizer(loader.dataset, keys[:-1], **kwargs)
    
    dims = [tmp_data[key].shape[-1] if tmp_data[key].ndim > 1 else 1 for key in normalizer.mean.keys()]
    splits = np.cumsum([0] + dims)
    splits = [(splits[i - 1], splits[i]) for i in range(1, len(splits))]
    
    def sample_fn(pmap: Optional[bool] = False, **sample_kwargs) -> Dict[str, np.ndarray]:
        batch = loader.sample(batch_size, **sample_kwargs)
        if normalize:
            batch = normalizer.normalize(batch)
            
        batch = {"traj_seq": np.concatenate([batch[key] for key in keys], axis=-1), "masks": batch["masks"]}
        
        if pmap:
            batch = shard(batch)

        return batch
    
    return sample_fn, (normalizer, splits)


def mlm_batch_sampler(loader: AntMLMDataLoader,
                      batch_size: int,
                      data_collator: "MLMDataCollator",
                      normalize: bool = False,
                      **kwargs):
    
    tmp_data, _, _ = loader.sample()
    keys = ('goals', 'observations', 'actions', loader.terminal_key)
    normalizer = Normalizer(loader.dataset, keys[:-1], **kwargs)

    dims = [tmp_data[key].shape[-1] if tmp_data[key].ndim > 1 else 1 for key in normalizer.mean.keys()]
    splits = np.cumsum([0] + dims)
    splits = [(splits[i - 1], splits[i]) for i in range(1, len(splits))]
    
    collator_fn = pad_shard_unpad(jax.pmap(data_collator.__call__), static_argnums=(3,), static_return=False)

    def sample_fn(pmap: Optional[bool] = False, rng: jp.ndarray = None, **sample_kwargs) -> Dict[str, np.ndarray]:
        batch1, batch2, nsp_labels = loader.sample(batch_size, **sample_kwargs)
        if normalize:
            batch1 = normalizer.normalize(batch1)
            batch2 = normalizer.normalize(batch2)

        if rng is not None:
            rng = flax.jax_utils.replicate(rng)

        batch1 = {"traj_seq": np.concatenate([batch1[key] for key in keys], axis=-1), "masks": batch1["masks"]}
        batch2 = {"traj_seq": np.concatenate([batch2[key] for key in keys], axis=-1), "masks": batch2["masks"]}
        batch = collator_fn(batch1, batch2, nsp_labels, rng)
        
        if pmap:
            batch = shard(batch)

        return batch
    
    return sample_fn, (normalizer, splits)


@flax.struct.dataclass
class MLMDataCollator:
    tokenizer: VQVAE
    params: flax.core.FrozenDict
    configs: ModelConfig
    seed: Optional[int] = 0

    def __call__(self, batch1: Dict[str, jp.ndarray], batch2: Dict[str, jp.ndarray], nsp_labels: jp.ndarray, rng: jp.ndarray = None) -> Dict[str, np.ndarray]:
        # x_enc1 = self.tokenizer.encode(**batch1, train=False)
        x_enc1 = self.tokenizer.apply(self.params, **batch1, train=False, method=self.tokenizer.encode)
        x_enc2 = self.tokenizer.apply(self.params, **batch2, train=False, method=self.tokenizer.encode)
        _, vq_info1 = self.tokenizer.apply(self.params, x_enc1, train=False, method=self.tokenizer.quantize)
        _, vq_info2 = self.tokenizer.apply(self.params, x_enc2, train=False, method=self.tokenizer.quantize)

        ids1 = vq_info1["indices"]
        ids2 = vq_info2["indices"]

        batch_size = ids1.shape[0]
        input_ids = jp.concatenate([jp.full((batch_size, 1), self.configs.cls_token), ids1, 
                                    jp.full((batch_size, 1), self.configs.sep_token), ids2], axis=1)
        
        nsp_ids_mask = jp.zeros_like(ids2) if self.configs.use_nsp else jp.ones_like(ids2)
        special_tokens_mask = jp.concatenate([jp.ones((batch_size, 1)),
                                              jp.zeros_like(ids1), 
                                              jp.ones((batch_size, 1)),
                                              nsp_ids_mask], axis=1, dtype=bool)
        
        type_ids = jp.concatenate([jp.zeros((batch_size, 2 + ids1.shape[1])),
                                   jp.ones((batch_size, ids2.shape[1]))], axis=1, dtype='i4')
        
        attn_masks = jp.ones_like(type_ids) if self.configs.use_nsp else 1 - type_ids.copy()
        
        input_ids, labels = self.modify_token(input_ids, special_tokens_mask, rng)
        return {"input_ids": input_ids.astype("i4"), 
                "type_ids": type_ids.astype("i4"),
                "labels": labels.astype("i4"), 
                "nsp_labels": nsp_labels.astype("i4"),
                "mask": attn_masks.astype("i4")[..., None]}

    def modify_token(self, input_ids: np.ndarray, special_tokens_mask: np.ndarray, rng: jp.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        labels = input_ids.copy()
        special_tokens_mask = special_tokens_mask.astype(bool)

        if rng is None:
            rng = jax.random.PRNGKey(self.seed)
        
        modi_rng, mask_rng, rand_rng, rand_ids_rng = jax.random.split(rng, 4)
        
        probs = jp.full(labels.shape, self.configs.modify_prob)
        probs = jp.where(special_tokens_mask, 0.0, probs)
        modifying_flags = jax.random.bernoulli(modi_rng, probs).astype(bool)

        masking_flags = jax.random.bernoulli(mask_rng, jp.full(labels.shape, self.configs.mask_prob)).astype(bool)
        masking_flags &= modifying_flags

        random_flags = jax.random.bernoulli(rand_rng, jp.full(labels.shape, self.configs.random_prob)).astype(bool)
        random_flags &= modifying_flags & ~masking_flags

        input_ids = jp.where(masking_flags, self.configs.mask_token, input_ids)
        random_tokens = jax.random.randint(rand_ids_rng, labels.shape, 0, self.configs.n_traj_tokens)
        input_ids = jp.where(random_flags, random_tokens, input_ids)

        labels = jp.where(modifying_flags, labels, -100)
        return input_ids, labels
    
    
if __name__ == "__main__":
    import optax
    from src.scripts.vae_prepare import prepare_config_dataset, prepare_states
    from src.scripts.prior_prepare import prepare_config_dataset
    from src.utils.context import load_state
    from src.common.configs import ModelConfig
    from pathlib import Path
    import pickle
    
    save_path = Path("/home/bluesun/PycharmProjects/tmp/BERTAP_v2/save/BERTAP_VAE-0615-1437")
    loader, prior_configs, vae_params, vae_configs = prepare_config_dataset(save_path, batch_size=4, n_epochs=10)
    collator = MLMDataCollator(VQVAE(vae_configs.model_config, training=False), vae_params, prior_configs.model_config)
    sampler_fn, (normalizer, splits) = mlm_batch_sampler(loader, 4, collator, normalize=True)

    import time
    for i in range(10):
        st = time.time()
        batch = sampler_fn()
        print(f"Time: {time.time() - st:.4f}s")
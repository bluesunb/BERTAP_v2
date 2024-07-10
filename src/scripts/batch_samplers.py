import flax.jax_utils
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
from functools import partial
from typing import Optional, Callable, Tuple, List, Dict, Any


def vae_batch_sampler(
    loader: AntDataLoader, 
    batch_size: int, 
    normalize: bool = True, 
    **kwargs
) -> Tuple[Callable[[], Dict[str, np.ndarray]], Tuple[Normalizer, List[Tuple[int, int]]]]:
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
    
    def sample_fn(pmap: bool = False, **sample_kwargs) -> Dict[str, np.ndarray]:
        batch = loader.sample(batch_size, **sample_kwargs)
        if normalize:
            batch = normalizer.normalize(batch)
            
        batch = {"traj_seq": np.concatenate([batch[key] for key in keys], axis=-1), "masks": batch["masks"]}

        if pmap:
            batch = shard(batch)

        return batch
    
    return sample_fn, (normalizer, splits)


def gpt_batch_sampler(
    loader: AntDataLoader, 
    batch_size: int,  
    data_collator: "GPTDataCollator",
    normalize: bool = False, 
    **kwargs
) -> Tuple[Callable[[], Dict[str, np.ndarray]], Tuple[Normalizer, List[Tuple[int, int]]]]:
    
    tmp_data = loader.sample()
    keys = ('goals', 'observations', 'actions', loader.terminal_key)
    normalizer = Normalizer(loader.dataset, keys[:-1], **kwargs)
    
    dims = [tmp_data[key].shape[-1] if tmp_data[key].ndim > 1 else 1 for key in normalizer.mean.keys()]
    splits = np.cumsum([0] + dims)
    splits = [(splits[i - 1], splits[i]) for i in range(1, len(splits))]
    
    collate_fn = pad_shard_unpad(jax.pmap(data_collator.__call__), static_argnums=(1,), static_return=False)

    def sample_fn(pmap: bool = False, return_conditioned: bool = False, rng: jp.ndarray = None, **sample_kwargs) -> Dict[str, np.ndarray]:
        batch = loader.sample(batch_size, **sample_kwargs)
        if normalize:
            batch = normalizer.normalize(batch)
            
        if rng is not None:
            rng = flax.jax_utils.replicate(rng)
        
        batch = {"traj_seq": np.concatenate([batch[key] for key in keys], axis=-1), "masks": batch["masks"]}
        condition = batch["traj_seq"][:, :1, :dims[0] + dims[1]]

        batch = collate_fn(batch, rng)
        batch["condition"] = condition
        
        if pmap:
            batch = shard(batch)
            
        return batch
    
    return sample_fn, (normalizer, splits)


def mlm_batch_sampler(
    loader: AntMLMDataLoader,
    batch_size: int,
    data_collator: "MLMDataCollator",
    normalize: bool = False,
    **kwargs
) -> Tuple[Callable[[], Dict[str, np.ndarray]], Tuple[Normalizer, List[Tuple[int, int]]]]:
    
    tmp_data, _, _ = loader.sample()
    keys = ('goals', 'observations', 'actions', loader.terminal_key)
    normalizer = Normalizer(loader.dataset, keys[:-1], **kwargs)

    dims = [tmp_data[key].shape[-1] if tmp_data[key].ndim > 1 else 1 for key in normalizer.mean.keys()]
    splits = np.cumsum([0] + dims)
    splits = [(splits[i - 1], splits[i]) for i in range(1, len(splits))]
    
    collator_fn = pad_shard_unpad(jax.pmap(data_collator.__call__), static_argnums=(5,), static_return=False)
    # collator_fn = data_collator.__call__

    def sample_fn(pmap: bool = False, rng: jp.ndarray = None, **sample_kwargs) -> Dict[str, np.ndarray]:
        batch1, batch2, nsp_labels = loader.sample(batch_size, **sample_kwargs)
        if normalize:
            batch1 = normalizer.normalize(batch1)
            batch2 = normalizer.normalize(batch2)

        if rng is not None:
            rng = flax.jax_utils.replicate(rng)

        batch1 = {"traj_seq": np.concatenate([batch1[key] for key in keys], axis=-1), "masks": batch1["masks"]}
        batch2 = {"traj_seq": np.concatenate([batch2[key] for key in keys], axis=-1), "masks": batch2["masks"]}

        condition1 = batch1["traj_seq"][:, :1, :dims[0] + dims[1]]
        condition2 = batch2["traj_seq"][:, :1, :dims[0] + dims[1]]

        batch = collator_fn(batch1, batch2, condition1, condition2, nsp_labels, rng)
        
        if pmap:
            batch = shard(batch)

        return batch
    
    return sample_fn, (normalizer, splits)


@flax.struct.dataclass
class GPTDataCollator:
    tokenizer: VQVAE
    params: flax.core.FrozenDict
    configs: ModelConfig = None
    seed: Optional[int] = 0
    
    def __call__(self, batch: Dict[str, jp.ndarray], rng: jp.ndarray = None) -> Dict[str, np.ndarray]:
        x_enc = self.tokenizer.apply(self.params, **batch, train=False, method=self.tokenizer.encode)
        _, vq_info = self.tokenizer.apply(self.params, x_enc, train=False, method=self.tokenizer.quantize)
        ids = vq_info["indices"]
        return {"input_ids": ids.astype("i4"), "labels": ids.astype("i4"), "mask": jp.ones_like(ids).astype("i4")[..., None]}


@flax.struct.dataclass
class MLMDataCollator:
    tokenizer: VQVAE
    params: flax.core.FrozenDict
    configs: ModelConfig
    seed: Optional[int] = 0

    def __post_init__(self):
        assert self.configs.modify_prob > 0.0, \
            "MLMDataCollator: modify_prob must be greater than 0.0, if not, n_labels will be 0 that will make nan error"

    def __call__(
        self, 
        batch1: Dict[str, jp.ndarray], 
        batch2: Dict[str, jp.ndarray], 
        condition1: jp.ndarray,
        condition2: jp.ndarray,
        nsp_labels: jp.ndarray, 
        rng: jp.ndarray = None
    ) -> Dict[str, np.ndarray]:
        
        x_enc1 = self.tokenizer.apply(self.params, **batch1, train=False, method=self.tokenizer.encode)
        _, vq_info1 = self.tokenizer.apply(self.params, x_enc1, train=False, method=self.tokenizer.quantize)
        ids1 = vq_info1["indices"]
        
        batch_size = ids1.shape[0]
        
        if self.configs.use_nsp:
            x_enc2 = self.tokenizer.apply(self.params, **batch2, train=False, method=self.tokenizer.encode)
            _, vq_info2 = self.tokenizer.apply(self.params, x_enc2, train=False, method=self.tokenizer.quantize)
            ids2 = vq_info2["indices"]
        else:
            x_enc2 = jp.empty((batch_size, 0, 0), dtype="f4")
            ids2 = jp.empty((batch_size, 0), dtype="i4")

        input_ids = jp.concatenate([jp.full((batch_size, 1), self.configs.cls_token), ids1,
                                    jp.full((batch_size, 1), self.configs.sep_token), ids2], axis=1)

        nsp_ids_mask = jp.zeros_like(ids2) if self.configs.use_nsp else jp.ones_like(ids2)
        special_tokens_mask = jp.concatenate([jp.ones((batch_size, 1)),
                                              jp.zeros_like(ids1),
                                              jp.ones((batch_size, 1)),
                                              nsp_ids_mask], axis=1, dtype=bool)

        # type_ids = jp.concatenate([jp.zeros((batch_size, 2 + ids1.shape[1])),
        #                            jp.ones((batch_size, ids2.shape[1]))], axis=1, dtype='i4')
        
        conditions = jp.concatenate([condition1.repeat(ids1.shape[1] + 2, axis=1),
                                     condition2.repeat(ids2.shape[1], axis=1)], axis=1)
        if self.configs.use_nsp:
            attn_masks = jp.ones_like(input_ids)
        else:
            attn_masks = jp.concatenate([jp.ones((batch_size, 2 + ids1.shape[1])),
                                         jp.zeros((batch_size, ids2.shape[1]))], axis=1)

        input_ids, labels = self.modify_token(input_ids, special_tokens_mask, rng)
        return {"input_ids": input_ids.astype("i4"),
                "conditions": conditions.astype("f4"),
                "labels": labels.astype("i4"),
                "nsp_labels": nsp_labels.astype("i4"),
                "mask": attn_masks.astype("i4")[..., None]}

    def modify_token(self, input_ids: jp.ndarray, special_tokens_mask: jp.ndarray, rng: jp.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
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

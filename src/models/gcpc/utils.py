import jax
import jax.numpy as jp


def make_random_mask(rng: jp.ndarray,
                     mask_excepts: jp.ndarray,
                     n_masks: int = 1) -> jp.ndarray:
    """
    Masking strategy such as:
    [0, ..., 1, 1, ..., 0, 1, 0, ...]   (1 for mask)
    """
    size = mask_excepts.shape
    n_masks = jp.broadcast_to(jp.asarray(n_masks, dtype=jp.int32), (size[0], 1))
    mask_seed = jax.random.uniform(rng, size)
    mask_seed = jp.where(mask_excepts, 0, mask_seed)
    cutoff = jp.take_along_axis(jp.sort(mask_seed, axis=-1), size[-1] - n_masks, axis=-1)
    mask = jp.asarray(mask_seed >= cutoff, dtype=jp.int32)
    # If the number of masks is greater than the number of unmasked tokens, then all tokens are masked:
    mask = jp.where(mask_excepts.sum(axis=-1, keepdims=True) > size[-1] - n_masks, 0, mask)
    return mask.astype(jp.float32)


# @partial(jax.jit, static_argnames=("min_unmasked", ))
def make_causal_mask(rng: jp.ndarray,
                     mask_excepts: jp.ndarray,
                     min_unmasked: int = 1) -> jp.ndarray:
    """
    Masking strategy such as:
    [0, ..., 0, 1, 1, 1, ..., 1]    (1 for mask)
    """
    size = mask_excepts.shape
    mask = jp.zeros(size, dtype=jp.int32)
    mask_seed = jax.random.uniform(rng, size)
    mask_seed = mask_seed.at[:, :min_unmasked].set(0)
    mask_seed = jp.where(mask_excepts, 0, mask_seed)
    mask_ids = mask_seed.argmax(axis=-1, keepdims=True)
    mask = mask.at[jp.arange(size[0])[:, None], mask_ids].set(1)
    mask = jp.where(mask_excepts.sum(axis=-1, keepdims=True) > size[-1] - min_unmasked, 0, mask)
    mask = mask.cumsum(axis=-1)
    mask = jp.where(jp.logical_and(mask > 0, mask_excepts < 1), 1, 0)   # make sure that the mask is not applied to the excepts
    return mask.astype(jp.float32)


# @partial(jax.jit, static_argnames=("n_masks", "latent_step"))
def make_subseq_mask(rng: jp.ndarray,
                     mask_excepts: jp.ndarray,
                     n_masks: int,
                     latent_step: int) -> jp.ndarray:
    """
    Masking strategy such as:
    [0, 0, ..., 1, ...(latent_steps)..., 0, 1, ...(latent_steps)..., 0]     (1 for mask)
    """
    size = (*mask_excepts.shape[:-1], mask_excepts.shape[-1] // latent_step)
    # mask = jp.zeros(size, dtype=jp.int32)
    n_masks = jp.broadcast_to(jp.asarray(n_masks, dtype=jp.int32), (size[0], 1))
    mask_excepts = mask_excepts.reshape((*mask_excepts.shape[:-1], -1, latent_step))
    mask_excepts = mask_excepts.all(axis=-1)
    mask_seed = jax.random.uniform(rng, size)
    mask_seed = jp.where(mask_excepts, 0, mask_seed)
    cutoff = jp.take_along_axis(jp.sort(mask_seed, axis=-1), size[-1] - n_masks, axis=-1)
    mask = jp.asarray(mask_seed >= cutoff, dtype=jp.int32)
    mask = jp.where(mask_excepts.sum(axis=-1, keepdims=True) > size[-1] - n_masks, 0, mask)
    mask = mask.repeat(latent_step, axis=-1)
    return mask.astype(jp.float32)


"""
Controller utilities for iterating over the 1D-scaled model state and
identifying \"alternative realities\" (high-lineage feature trajectories).

This module is intentionally minimal and JAX-friendly:
- indexing is expressed as pure functions over JAX arrays
- iteration uses lax.scan / lax.fori_loop / vmap where appropriate
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp


@dataclass
class FlatController:
    """
    Controller over a flat (1D) model buffer.

    `segment_lengths` usually comes from a DB/param controller (e.g. DB_PARAM_CONTROLLER)
    and defines how many scalars belong to each logical parameter.
    """
    segment_lengths: jnp.ndarray  # shape: (P,)

    @property
    def cumsum(self) -> jnp.ndarray:
        # [0, l0, l0+l1, ...]
        zeros = jnp.array([0], dtype=self.segment_lengths.dtype)
        return jnp.concatenate([zeros, jnp.cumsum(self.segment_lengths)])

    def index_of(self, param_idx: int) -> Tuple[int, int]:
        """
        Return (start, length) in the flat buffer for logical param_idx.
        """
        start = self.cumsum[param_idx]
        length = self.segment_lengths[param_idx]
        return int(start), int(length)

    def slice_param(self, flat_model: jnp.ndarray, param_idx: int) -> jnp.ndarray:
        """
        Slice out the 1D segment for logical parameter `param_idx`.
        """
        start, length = self.index_of(param_idx)
        return flat_model[start : start + length]


def iterate_flat_model(
    flat_model: jnp.ndarray,
    controller: FlatController,
    step_fn: Callable[[jnp.ndarray, int], jnp.ndarray],
) -> jnp.ndarray:
    """
    Iterate over the flat 1D model using a controller and a pure step function.

    Parameters
    ----------
    flat_model : (N,)
        Current flat state (e.g. scaled DB or model buffer).
    controller : FlatController
        Provides (start, length) for each logical parameter.
    step_fn : callable (segment, idx) -> new_segment
        Called for each parameter segment; should be side-effect free.

    Returns
    -------
    new_flat_model : (N,)
        Model after applying `step_fn` to each segment.
    """
    num_params = controller.segment_lengths.shape[0]

    def body(i, model):
        seg = controller.slice_param(model, i)
        new_seg = step_fn(seg, i)
        start, length = controller.index_of(i)
        return jax.lax.dynamic_update_slice(model, new_seg, (start,))

    new_flat = jax.lax.fori_loop(0, num_params, body, flat_model)
    return new_flat


def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Cosine similarity between two vectors or matching batches.
    """
    a_flat = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(1, -1)
    b_flat = b.reshape(b.shape[0], -1) if b.ndim > 1 else b.reshape(1, -1)
    dot = jnp.sum(a_flat * b_flat, axis=-1)
    na = jnp.linalg.norm(a_flat, axis=-1) + eps
    nb = jnp.linalg.norm(b_flat, axis=-1) + eps
    return dot / (na * nb)


def recognize_alternative_realities(
    in_history: jnp.ndarray,
    out_history: jnp.ndarray,
    *,
    threshold: float = 0.99,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Recognize \"alternative realities\" based on in/out feature lineage.

    Parameters
    ----------
    in_history : (T, D_in)
        Time-series of input features for one logical item.
    out_history : (T, D_out)
        Time-series of output features for the same item.
    threshold : float
        Minimal cosine similarity to mark lineage (default: 0.99).

    Returns
    -------
    best_idx : (T,)
        For each t, index of the most similar past step (0..t).
    mask_alt : (T,)
        Boolean mask where similarity >= threshold (high-lineage alternative realities).
    """

    def body(carry, t):
        # compare current step to all past (including itself)
        h_in = in_history[: t + 1]
        h_out = out_history[: t + 1]

        sim_in = cosine_similarity(in_history[t], h_in)
        sim_out = cosine_similarity(out_history[t], h_out)

        # simple combination: average in/out similarity
        sim = 0.5 * (sim_in + sim_out)

        idx = jnp.argmax(sim)
        best_sim = sim[idx]
        is_alt = best_sim >= threshold
        return carry, (idx, is_alt)

    _, (best_idx, mask_alt) = jax.lax.scan(
        body,
        None,
        jnp.arange(in_history.shape[0]),
    )
    return best_idx, mask_alt


"""
Training iteration: single step with optional optimizer and gradient handling.
Stable pattern for JAX; optional ejkernel config can drive kernel/attention opts inside loss.
"""
import jax


def run_training_step(
    params,
    batch,
    loss_fn,
    *,
    opt_state=None,
    opt_update=None,
):
    """
    One training step: compute loss, gradients, and optionally update params.

    Parameters
    ----------
    params : pytree
        Model parameters.
    batch : pytree
        Batch of inputs (e.g. (x, y) or (inputs, targets)).
    loss_fn : callable (params, batch) -> scalar
        Loss to minimize.
    opt_state : pytree, optional
        Optimizer state (e.g. Adam); if None, only loss and grads are returned.
    opt_update : callable (opt_state, grads) -> (new_opt_state, new_params), optional
        Optimizer update; required if opt_state is not None.

    Returns
    -------
    loss : float
        Current step loss.
    grads : pytree
        Gradients w.r.t. params.
    new_params : pytree, optional
        Updated params (only if opt_state and opt_update provided).
    new_opt_state : pytree, optional
        Updated optimizer state (only if provided).
    """
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    out = (loss, grads)
    if opt_state is not None and opt_update is not None:
        new_opt_state, new_params = opt_update(opt_state, grads, params)
        out = (loss, grads, new_params, new_opt_state)
    return out

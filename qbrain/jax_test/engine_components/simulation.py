"""
Simulation iteration: JAX-stable step loop for time evolution.
Uses lax.scan so the full trajectory is traceable and differentiable (no Python for-loop over steps).
Optional: ejkernel can be used inside step_fn for attention/kernel ops.
"""
import jax


def run_simulation_scan(
    carry,
    num_steps,
    step_fn,
    *,
    unroll=1,
):
    """
    Run a fixed number of simulation steps in a JAX-stable way.

    Parameters
    ----------
    carry : pytree
        Initial state (e.g. db state, time index).
    num_steps : int
        Number of steps (must be concrete for jit).
    step_fn : callable (carry -> (new_carry, output))
        One-step function. Called as (carry,) -> (carry, out); output is stacked.
    unroll : int
        Scan unroll factor (1 = no unroll; 4 is common for small loops).

    Returns
    -------
    final_carry : pytree
        State after num_steps.
    stacked_outputs : pytree of (num_steps, ...)
        Per-step outputs from step_fn.
    """
    def scan_body(c, _):
        c, out = step_fn(c)
        return c, out

    final_carry, stacked_outputs = jax.lax.scan(
        scan_body,
        carry,
        None,
        length=num_steps,
        unroll=unroll,
    )
    return final_carry, stacked_outputs

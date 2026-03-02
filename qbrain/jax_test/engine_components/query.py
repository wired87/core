"""
Query / iteration over time or sequence: JAX-stable scan for inference or rollout.
Ensures deterministic, traceable query over steps (e.g. temporal GNN, rollout).
"""
import jax


def run_query_scan(
    init_carry,
    inputs,
    query_fn,
    *,
    unroll=1,
):
    """
    Run a query (inference/rollout) over a sequence in a JAX-stable way.

    Parameters
    ----------
    init_carry : pytree
        Initial carry (e.g. hidden state, cache).
    inputs : pytree of (T, ...)
        Sequence to step over (time or batch dimension first).
    query_fn : callable (carry, inp) -> (new_carry, out)
        Per-step query; e.g. one temporal model step.
    unroll : int
        lax.scan unroll factor.

    Returns
    -------
    final_carry : pytree
        Carry after processing the full sequence.
    outputs : pytree of (T, ...)
        Per-step outputs.
    """
    final_carry, outputs = jax.lax.scan(
        query_fn,
        init_carry,
        inputs,
        unroll=unroll,
    )
    return final_carry, outputs

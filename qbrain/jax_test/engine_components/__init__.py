# Engine components: stable iteration for simulation, training, and query.
# Designed for JAX traceability; optional ejkernel integration for kernel/attention ops.
from engine_components.simulation import run_simulation_scan
from engine_components.training import run_training_step
from engine_components.query import run_query_scan
from engine_components.ejkernel_opt import is_ejkernel_available, get_ejkernel_config

__all__ = [
    "run_simulation_scan",
    "run_training_step",
    "run_query_scan",
    "is_ejkernel_available",
    "get_ejkernel_config",
]

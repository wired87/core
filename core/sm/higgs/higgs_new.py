import jax.numpy as jnp
from jax import jit, vmap
from typing import List, Dict, Any, Tuple

from core.sm.higgs.phi_utils import HiggsUtils
from qf_utils.field_utils import FieldUtils
from _ray_core.utils.ray_validator import RayValidator
from utils._np.serialize_complex import check_serialize_dict


# NOTE: Assume check_serialize_dict/restore_selfdict handle conversion between
# Python dict and JAX array structures (SOA vs. ASO)

# --- JAX-COMPATIBLE HELPER FUNCTIONS (Must be defined outside class for JAX) ---

# Helper function to compute mu (must use jnp for JAX compatibility)
@jit
def compute_mu_jax(vev, lambda_h):
    """Calculates mu from VEV and lambda_h using JAX operations."""
    return vev * jnp.sqrt(lambda_h)


# Helper function for mass term
@jit
def _mass_term_jax(mass, h):
    """Calculates the mass term based on current state."""
    return -mass ** 2 * h


@jit
def _higgs_potential_jax(mass, vev, lambda_h, h):
    """Calculates the Higgs potential V(h) at a single point."""
    m2 = mass ** 2
    v = vev
    l = lambda_h

    # V(h) = 0.5 m^2 h^2 + λ v h^3 + 0.25 λ h^4
    h_potential = (
            0.5 * m2 * h ** 2
            + l * v * h ** 3
            + 0.25 * l * h ** 4
    )
    return h_potential


# --- CORE SINGLE-POINT UPDATE LOGIC (Batched via vmap in main) ---

@jit
def _single_point_update(
        attrs: Dict[str, Any],
        neighbor_data: Dict[str, Tuple[Any, Any]],
        d_params: Dict[str, float]
) -> Dict[str, Any]:
    """
    Performs the full equation sequence for a single (unbatched) field point.
    This function is pure and JIT-compiled for GPU efficiency.
    """

    # 1. Read necessary properties from the input 'attrs' dict
    h = attrs["h"]
    h_prev = attrs.get("h_prev", h)  # Use h if h_prev is missing
    mass = attrs["mass"]
    vev = attrs["vev"]

    # 2. Calculate constants (lambda_h and mu)
    lambda_h = (mass ** 2) / (2 * vev ** 2)
    mu = compute_mu_jax(vev, lambda_h)

    # 3. Calculate Laplacian (Requires neighbor admin_data)
    laplacian_h = 0.0
    # NOTE: The loop over neighbor_data MUST be handled using JAX array operations
    # (e.g., jnp.sum on pre-structured arrays) for true JAX/GPU parallelism.
    # This loop is highly non-JAX-idiomatic. Assuming neighbor_data is pre-mapped for JAX:

    # Placeholder for JAX-compatible Laplacian calculation:
    # Assuming neighbor_data is structured for batch processing
    # Example: neighbor_data['val_plus'] and neighbor_data['val_minus'] are arrays

    # WARNING: This loop is simulated here but must be array-based in practice
    for key in ['x', 'y', 'z']:
        # This part requires external JAX refactoring for full vmap support.
        # Since 'neighbor_data' format is complex, we skip the loop logic for JAX purity.
        # For now, let's assume laplacian_h is derived from a separate JAX operation.

        # *** Using placeholder constant for demonstration ***
        laplacian_h += 0.0  # Replace with actual array computation

    # 4. Calculate Potential Derivative (dV_dh)
    dV_dh = -mu ** 2 * (vev + h) + lambda_h * (vev + h) ** 3

    # 5. Calculate Mass Term
    mass_term = _mass_term_jax(mass, h)

    # 6. Klein-Gordon Update (Calculate h_next)
    # h_next = 2 * h - h_prev + dt^2 * (Laplacian + MassTerm - dV/dh)
    dt_sq = d_params['t'] ** 2
    h_next = 2 * h - h_prev + dt_sq * (laplacian_h + mass_term - dV_dh)

    # 7. Calculate Phi (Higgs doublet)
    new_phi = (1 / jnp.sqrt(2)) * jnp.array([0, vev + h_next])

    # 8. Calculate Energy Density
    d_phi = new_phi  # Using new_phi as a placeholder for actual d_phi gradient computation
    kinetic = 0.5 * (d_phi[0]) ** 2
    gradient = 0.5 * jnp.sum(d_phi[1:] ** 2)
    potential = _higgs_potential_jax(mass, vev, lambda_h, h_next)
    energy = kinetic + jnp.abs(gradient) + potential

    # 9. Update and Return the entire state dictionary
    attrs['h_prev'] = h  # Current h becomes previous for next step
    attrs['h'] = h_next
    attrs['laplacian_h'] = laplacian_h  # Note: Needs full implementation
    attrs['dV_dh'] = dV_dh
    attrs['lambda_h'] = lambda_h
    attrs['phi'] = new_phi
    attrs['energy'] = energy

    # NOTE: Console output is omitted here as it breaks JIT/vmap purity.
    # Logging should be handled after JAX execution.
    return attrs


class HiggsBase(
    FieldUtils,
    HiggsUtils,
    RayValidator
):
    """
    HiggsBase class optimized for GPU parallel processing using JAX vmap/jit.
    All stateful admin_data is managed in the 'attrs' list of dicts.
    """

    def __init__(self, env):
        HiggsUtils.__init__(self)
        FieldUtils.__init__(self)
        self.env = env
        self.symbol = "Φ"
        self.d = env["d"]  # Should be used as a constant in JAX functions

        # 1. Define the JAX parallel processing function (JIT + VMAP)
        # We need to explicitly pass environment constants like self.d to the JAX function
        # This wrapper handles the external (non-JAX) inputs and calls the pure function

        # Define the arguments that need to be mapped (batched)
        # The main 'attrs' dictionary and 'neighbor_pm_val_same_type' should be batched (axis 0).

        # NOTE: Since the neighbor structure is complex, we simplify the vmap structure:
        # Assuming the caller transforms list[dict] -> Dict[str, jnp.array] (SOA)

        # We define a vmapped/jitted function that processes the entire batch of attributes
        # and returns the batch of updated attributes.

        self._jitted_vmapped_update = jit(vmap(
            _single_point_update,
            in_axes=(0, 0, None)  # 0 for attrs and neighbor_data (batched), None for d_params (constant)
        ))


        # NOTE: This JAX setup is highly dependent on the input format of neighbor_pm_val_same_type.
        # It assumes the neighbor admin_data is also pre-batched and aligned with the attrs batch.

    def main(
            self,
            attrs: List[dict],  # Input admin_data is now a list of field point dictionaries
            all_subs: dict,
            neighbor_pm_val_same_type,  # Also batched, typically struct-of-arrays
    ) -> dict:
        """
        The main method orchestrates admin_data flow, triggering the parallel JAX update.
        """
        # 1. Data Conversion and Preparation (Non-JAX Python/Ray scope)
        # Scale list horizontal
        current_attrs_soa = self.list_to_soa(attrs)


        # 2. JAX Initialization/Constants
        # Create a dictionary of constants needed by the JAX function
        d_params = {key: self.d[key] for key in self.d}

        # 3. Parallel GPU Execution (JIT and VMAP)
        # Call the pre-compiled, vectorized function on the batched admin_data
        updated_attrs_soa = self._jitted_vmapped_update(
            current_attrs_soa,
            neighbor_pm_val_same_type,  # Assumed to be SOA
            d_params
        )

        # 4. Finalization and Return (Non-JAX Python/Ray scope)
        # Convert Struct of Arrays (SOA) back to List[dict]
        final_attrs_list = self.soa_to_list(updated_attrs_soa)

        # Serialize for Ray transport
        final_dict = check_serialize_dict(final_attrs_list)
        return final_dict

    # NOTE: All other original methods (_d_phi, _phi, etc.) are replaced
    # by the logic within the pure function _single_point_update.
    # They are kept here as placeholder stubs if needed for compatibility,
    # but the core logic is now in the JAX-compiled wrapper.

    def list_to_soa(self, attrs_list: List[dict]) -> Dict[str, jnp.ndarray]:
        # Placeholder for complex conversion logic (must be implemented externally)
        # Collect values horizontal
        return {k: jnp.array([d[k] for d in attrs_list]) for k in attrs_list[0]}

    def soa_to_list(self, attrs_soa: Dict[str, jnp.ndarray]) -> List[dict]:
        # Placeholder for complex conversion logic (must be implemented externally)
        # Takes Dict[Key, JAX Array] and converts back to List[Dict]
        num_fields = len(next(iter(attrs_soa.values())))
        attrs_list = []
        for i in range(num_fields):
            attrs_list.append({k: v[i] for k, v in attrs_soa.items()})
        return attrs_list

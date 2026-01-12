import jax.numpy as jnp
from jax import jit, vmap
from typing import List, Dict, Any, Tuple

from core.sm.fermion.ferm_utils import FermUtils
from _ray_core.utils.ray_validator import RayValidator


@jit
def _psi_bar_jax(psi, is_quark):
    """Calculates psi_bar for a single field point."""
    # NOTE: This implementation requires the Gamma matrices (gamma0, etc.)
    # to be passed as constants or globals if they are used here.
    # Assuming the core operation logic is available for JAX.
    return psi.conjugate().T  # Placeholder simplified logic


@jit
def _dirac_process_jax(psi, prev, dpsi, mass, d_params, all_subs_data) -> Tuple[Any, Any]:
    """
    Performs the full Dirac equation update for a single field point.

    This function combines kinetic term (dpsi), mass term, and coupling term
    (from all_subs_data) to calculate the next field state (psi_next).
    """

    # 1. Calculate Mass Term (i * m * psi)
    # Assuming 'i' (imaginary unit) is available as jnp.complex64(0+1j)
    i = jnp.complex64(0 + 1j)
    mass_term = -i * mass * psi

    # 2. Combine all terms (Kinetic + Mass + Interaction)
    # NOTE: The complex interactions/coupling calculation (W/Z/Gluon/Yukawa)
    # must be pre-calculated and included in 'all_subs_data' as structured arrays
    # for JAX/vmap compatibility.

    # Placeholder for the complex interaction term (J_mu * Gauge)
    interaction_term = all_subs_data.get('coupling_sum', 0.0)

    # 3. Time evolution based on Dirac equation (Simplified Klein-Gordon approach)
    # The Dirac result is the time derivative term (∂tψ)
    # Assuming dpsi[0] is the time derivative term
    dpsi_dt = dpsi[0]

    # Update ψ(t+Δt) = ψ(t) + Δt * ∂tψ
    # The Dirac process should output the time derivative:
    dirac_result = (dpsi_dt + interaction_term) + mass_term

    # Time evolution step
    psi_next = psi + d_params['t'] * dirac_result

    return psi_next, dirac_result  # Return both for subsequent checks/storage


# --- CORE SINGLE-POINT UPDATE LOGIC (Batched via vmap in main) ---

@jit
def _single_point_update(
        attrs: Dict[str, Any],
        all_subs_data: Dict[str, Any],
        d_params: Dict[str, float]
) -> Dict[str, Any]:
    """
    Performs the full equation sequence for a single (unbatched) field point.
    This function is pure and JIT-compiled for GPU efficiency.
    """

    # 1. Read input state variables
    psi = attrs["psi"]
    mass = attrs["mass"]
    is_quark = attrs["is_quark"]

    # 2. Calculate dependent variables
    attrs['psi_bar'] = _psi_bar_jax(psi, is_quark)

    # 3. Calculate Dpsi (Kinetic/Spatial Gradient Term)
    # NOTE: In a true JAX implementation, dpsi (spatial derivatives) are
    # typically calculated using jnp.roll/convolve on the whole grid,
    # not within the single-point vmap.
    # Placeholder: assume dpsi is retrieved from pre-calculated admin_data
    dpsi = attrs.get('dpsi_kin', jnp.zeros_like(psi))

    # 4. Perform Dirac Update (Time Evolution)
    psi_next, dirac_result = _dirac_process_jax(
        psi,
        attrs.get("prev", psi),
        dpsi,
        mass,
        d_params,
        all_subs_data  # Interaction terms included here
    )

    # 5. Update and Return the entire state dictionary
    attrs['prev'] = psi  # Current psi becomes previous for next step
    attrs['psi'] = psi_next
    attrs['dpsi'] = dpsi
    attrs['dirac_result'] = dirac_result

    return attrs


class FermionBase(
    FermUtils,
    RayValidator
):
    """
    FermionBase class optimized for GPU parallel processing using JAX vmap/jit.
    All stateful admin_data is managed via the 'attrs' list of dicts.
    """

    def __init__(self, env):
        FermUtils.__init__(self)
        self.env = env
        self.d = env["d"]  # Should be used as a constant in JAX functions
        self.quark_type_coupling_check = ["up", "charm", "top"]

        # 1. Define the JAX parallel processing function (JIT + VMAP)
        # in_axes: (0: attrs batch, 0: all_subs_data batch, None: constants d_params)
        self._jitted_vmapped_update = jit(vmap(
            _single_point_update,
            in_axes=(0, 0, None)
        ))

        # Placeholder for external SOA conversion utility
        # NOTE: You need to ensure the caller provides 'all_subs' and 'neighbor_pm_val_same_type'
        # as a JAX struct-of-arrays (SOA) aligned with 'attrs'.

    def main(
            self,
            attrs: List[dict],  # Input admin_data is now a list of field point dictionaries
            all_subs: dict,  # Assumed to contain structured neighbor admin_data, converted externally
            neighbor_pm_val_same_type,
            **kwargs
    ) -> dict:
        """
        The main method orchestrates admin_data flow, triggering the parallel JAX update.
        """

        # 1. Data Conversion (Non-JAX Python/Ray scope)
        # Convert List[dict] -> Struct of Arrays (SOA: Dict[str, jnp.array])
        current_attrs_soa = self.list_to_soa(attrs)

        # 2. Aggregate and Convert Neighbor/Interaction Data (Critical Step)
        # NOTE: This placeholder assumes 'all_subs' and 'neighbor_pm_val_same_type'
        # are combined into a single, batched SOA structure aligned with the attrs batch.

        # Placeholder: Assuming the caller provides a pre-converted structure
        # containing all necessary interaction terms (gauge, yukawa) as SOA.
        all_subs_soa = self._convert_subscribers_to_soa(all_subs)

        # 3. JAX Initialization/Constants
        # Create a dictionary of constants (like time step dt) needed by the JAX function
        d_params = {key: self.d[key] for key in self.d}

        # 4. Parallel GPU Execution (JIT and VMAP)
        updated_attrs_soa = self._jitted_vmapped_update(
            current_attrs_soa,
            all_subs_soa,
            d_params
        )

        # 5. Finalization and Return (Non-JAX Python/Ray scope)
        # Convert Struct of Arrays (SOA) back to List[dict]
        final_attrs_list = self.soa_to_list(updated_attrs_soa)

        # Serialize for Ray transport
        print(">>>>>>>>>>>>>")
    # --- SOA CONVERSION HELPERS ---

    def list_to_soa(self, attrs_list: List[dict]) -> Dict[str, jnp.ndarray]:
        """Converts List[Dict] to Struct-of-Arrays (SOA) for JAX batching."""
        if not attrs_list:
            return {}
        # Simple implementation assuming all nested values are already compatible JAX types
        return {k: jnp.array([d[k] for d in attrs_list]) for k in attrs_list[0]}

    def soa_to_list(self, attrs_soa: Dict[str, jnp.ndarray]) -> List[dict]:
        """Converts Struct-of-Arrays (SOA) back to List[Dict]."""
        if not attrs_soa:
            return []
        num_fields = len(next(iter(attrs_soa.values())))
        attrs_list = []
        for i in range(num_fields):
            attrs_list.append({k: v[i] for k, v in attrs_soa.items()})
        return attrs_list

    def _convert_subscribers_to_soa(self, all_subs: dict) -> Dict[str, jnp.ndarray]:
        """
        PLACEHOLDER: Must implement logic to extract critical gauge/higgs neighbor
        admin_data from the complex 'all_subs' dictionary and convert it into a
        simple SOA array structure that aligns with the field point batch.
        """
        # For demonstration, returning a placeholder dictionary of zero arrays
        return {
            'coupling_sum': jnp.zeros(len(self.list_to_soa(all_subs).values()), dtype=jnp.complex64)
        }

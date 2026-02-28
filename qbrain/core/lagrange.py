import jax.numpy as jnp
from jax import jit
from typing import Tuple, List, Dict, Any  # Used for type hints, as requested

from qbrain.utils._jax.arsenal.FERMION import dmu_psi, psi_bar, gterm, yterm, dirac, psi
from qbrain.utils._jax.arsenal.GAUGE import dmuG, dmu_fmunu, fmunu, gg_coupling, gf_coupling, j_nu, field_value
from qbrain.utils._jax.arsenal.HIGGS import dmu_h, lambda_H, dV_dh, h


# Assume all helper functions (dmu_h, dirac, h, fmunu, etc.)
# from higgs.py, FERMION.py, and GAUGE.py are available in the scope.

@jit
def sm(
        # --- State Arrays ---
        h_grid: jnp.ndarray,
        prev_h_grid: jnp.ndarray,
        psi_grid: jnp.ndarray,
        prev_psi_grid: jnp.ndarray,
        gauge_grid: jnp.ndarray,  # A_mu field value
        prev_gauge_grid: jnp.ndarray,

        # --- Constants & Geometry ---
        dt: float,
        d: float,
        gamma_matrices: jnp.ndarray,
        gamma0_inv: jnp.ndarray,
        pm_axes: Tuple[List[Tuple], List[Tuple]],  # Shift directions for derivatives

        # --- Couplings & Masses ---
        mass_h: float,
        mass_f: float,
        vev: float,
        y_f: float,  # Fermion Yukawa coupling
        g_c: float,  # Gauge coupling constant
        ckm_struct: Dict[str, float],  # CKM parameters for Fermion doublets

        # --- Pre-calculated/Neighbor Data (Tuple format matching internal functions) ---
        gauge_neighbors_gterm: Any,  # for FERMION.py::gterm
        gauge_neighbors_gg: Any,  # for GAUGE.py::gg_coupling
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Orchestrates one complete, optimized time evolution step for the coupled SM fields.
    This function defines the final, fixed interaction path.

    The entire sequence is JIT-compiled for GPU efficiency.
    """

    # --- 1. KINETIC TERMS (Derivatives: d/dt and d/dx) ---
    # These are independent and can be calculated first.
    dmu_h_res = dmu_h(h_grid, prev_h_grid, d, dt, pm_axes)
    dmu_psi_res = dmu_psi(psi_grid, prev_psi_grid, d, dt, pm_axes)
    dmu_gauge_res = dmuG(gauge_grid, prev_gauge_grid, d, dt, pm_axes)

    # --- 2. GAUGE FIELD STRENGTH & CURRENTS ---
    # F^munu is needed before its derivative d_mu F^munu.
    fmunu_tensor = fmunu(dmu_gauge_res)

    # d_mu F^munu (derivative of the field strength)
    dmu_fmunu_res = dmu_fmunu(fmunu_tensor, fmunu_tensor, d, dt, pm_axes)  # prev_fmunu is fmunu for simplicity

    # Coupling terms (J_nu components)
    gg_current = gg_coupling(g_c, gauge_grid, gauge_neighbors_gg)

    # Placeholder for Fermion current calculation (needs psi_bar, gamma, T-matrices, etc.)
    # Note: gf_coupling expects a tuple/list of neighbor admin_data (psi, psi_bar, etc.)
    # We use psi_bar here for the current calculation
    psi_bar_res = psi_bar(psi_grid, gamma_matrices)

    # Simplified gf_coupling call
    gf_current = gf_coupling(
        neighbor_f=(psi_grid, psi_bar_res, gauge_grid, gauge_grid),  # Simplified neighbor structure
        field_value=gauge_grid,
        g=g_c,
    )

    # Total current j_nu = J_fermion + J_gauge_self
    j_nu_total = j_nu(gg_current, gf_current)

    # --- 3. FIELD COUPLINGS (Interaction Rules) ---
    # The terms coupling different fields are calculated here.

    # Fermion-Gauge Coupling (gterm)
    gterm_res = gterm(psi_grid, 0, gauge_neighbors_gterm)  # i=0 placeholder

    # Fermion-Higgs Coupling (yterm)
    yterm_res = yterm(y_f, psi_grid, psi_bar_res, h_grid)

    # --- 4. NEW FIELD EVOLUTION (Final Updates) ---

    # A. New Gauge Field (A_mu)
    new_gauge_grid = field_value(dt, dmu_fmunu_res, j_nu_total, gauge_grid)

    # B. New Fermion Field (Psi)
    dirac_res = dirac(
        psi=psi_grid,
        dmu=jnp.stack(dmu_psi_res),  # Stack derivative list back into array
        mass=mass_f,
        gterm=gterm_res,
        yterm=yterm_res,
        gamma=gamma_matrices,
        gamma0_inv=gamma0_inv,
        i=0  # Index i placeholder
    )
    new_psi_grid = psi(psi_grid, dt, dirac_res)

    # C. New Higgs Field (h)
    lambda_H_val = lambda_H(mass_h, vev)
    dV_dh_res = dV_dh(vev, lambda_H_val, h_grid)
    # laplacian_h_res = laplacian_h(h_grid, pm_axes, d, dt) # simplified here

    new_h_grid = h(
        h=h_grid,
        mass=mass_h,
        prev=prev_h_grid,
        laplacian_h=jnp.sum(jnp.stack(dmu_h_res), axis=0),  # Using derivative sum as Laplacian proxy
        dV_dh=dV_dh_res,
        dt=dt
    )

    return new_h_grid, new_psi_grid, new_gauge_grid
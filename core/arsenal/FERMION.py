# -----------------------------------------------------------------------------
# FERMION DYNAMICS - ATOMIC PHYSICS EQUATIONS
# -----------------------------------------------------------------------------

def calc_psi_bar(psi, gamma_0):
    # Dirac Adjoint: ψ_bar = ψ†γ⁰
    psi_bar = psi.conj().T @ gamma_0
    return psi_bar

def calc_yukawa_coupling(y, psi_bar, psi, neighbor_h):
    # Yukawa interaction term: -y * H * (ψ_bar @ ψ)
    yukawa_coupling = -y * neighbor_h * (psi_bar @ psi)
    return yukawa_coupling

def calc_yterm(yukawa_coupling_sum):
    # Total Yukawa contribution collected from neighbor interactions
    yterm = yukawa_coupling_sum
    return yterm

def calc_spatial_diff(field_forward, field_backward, d):
    # Spatial central difference: (ψ_{i+1} - ψ_{i-1}) / 2d
    spatial_diff = (field_forward - field_backward) / (2.0 * d)
    return spatial_diff

def calc_time_diff(psi, prev_psi, dt):
    # Temporal backward difference: (ψ_t - ψ_{t-dt}) / dt
    time_diff = (psi - prev_psi) / dt
    return time_diff

def calc_dirac_kinetic_component(gamma_mu, dmu_psi):
    # Kinetic component of Dirac equation: γμ @ ∂μψ
    dirac_kinetic_component = gamma_mu @ dmu_psi
    return dirac_kinetic_component

def calc_mass_term(_mass, psi):
    # Fermion mass contribution: -i * m * ψ
    mass_term = -1j * _mass * psi
    return mass_term

def calc_gterm_mu(i_imag, g, field_value, T, psi):
    # Gauge coupling component: -i * g * Aμ * (T @ ψ)
    gterm_mu = -i_imag * g * field_value * (T @ psi)
    return gterm_mu

def calc_gterm(gterm_mu_sum):
    # Total gauge interaction collected across dimensions and neighbors
    gterm = gterm_mu_sum
    return gterm

def calc_dirac(gamma0_inv, dirac_kinetic_sum, gterm, yterm, mass_term):
    # Dirac equation evolution: -γ⁰_inv @ (Σ(γμ∂μψ) + gterm + yterm + mass_term)
    dirac = -gamma0_inv @ (dirac_kinetic_sum + gterm + yterm + mass_term)
    return dirac

def calc_psi(psi, dt, dirac):
    # First-order Euler update for spinor field: ψ_new = ψ + dt * ∂tψ
    psi = psi + dt * dirac
    return psi

def calc_gauss(x, mu, sigma):
    # Gaussian distribution for wave packet initialization
    gauss = jnp.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return gauss

def calc_ckm_component(neighbor_psi, ckm_val):
    # CKM matrix rotation component for quark flavor mixing
    ckm_component = neighbor_psi * ckm_val
    return ckm_component

def calc_quark_doublet(psi, ckm_component_sum):
    # Construction of the SU(2) quark doublet: [ψ_up, Σ(V_ud * ψ_down)]
    quark_doublet = jnp.stack([psi, ckm_component_sum], axis=1)
    return quark_doublet
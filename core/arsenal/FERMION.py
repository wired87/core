# --------------------
# yterm (Yukawa Coupling)
# --------------------
from jax.numpy import vdot, stack, exp


def calc_yukawa_term(y, psi, psi_bar, neighbor_h):
    """
    Core Yukawa term calculation for single neighbor.
    """
    # Eq: -y * neighbor_h * vdot(psi_bar, psi)
    return -y * neighbor_h * vdot(psi_bar, psi)


def calc_yterm(y, psi, psi_bar, h):
    """
    Time evolution (for sim)
    Uses Kinetic derivation & coupling term of neighbors
    """
    # Programmatic intent: Sum of yukawa terms across neighbors
    return sum(calc_yukawa_term(y, psi, psi_bar, h))


# --------------------
# psi_bar (Conjugated Spinor)
# --------------------

def calc_psi_bar(psi, gamma):
    # Eq: psi.conj().T @ gamma[0]
    return psi.conj().T @ gamma[0]


# --------------------
# Derivatives (dmu_psi)
# --------------------

def calc_spatial_diff(field_forward, field_backward, d_space):
    """Berechnet (Psi_{i+1} - Psi_{i-1}) / (2.0 * d)"""
    return (field_forward - field_backward) / (2.0 * d_space)


def calc_time_diff(field_current, field_prev, d_time):
    """Berechnet (Psi(t) - Psi(t-dt)) / dt"""
    return (field_current - field_prev) / d_time


def calc_dmu_psi(psi, prev_psi, d, dt, p_axes, m_axes):
    """
    Calculates derivative vector.
    In production, extractor will map 'roll' to a shift-operator.
    """
    # dmu[0] = calc_time_diff(...)
    # dmu[1:] = calc_spatial_diff(roll(psi, p), roll(psi, m), d)
    # This structure is handled as a list of equations in the pathway
    return [calc_time_diff(psi, prev_psi, dt), calc_spatial_diff(psi, psi, d)]


# --------------------
# dirac_process
# --------------------

def calc_dirac_spatial_component(gamma_item, dmu_mu):
    # Eq: γ^i @ d_mu
    return gamma_item @ dmu_mu


def calc_mass_term(mass, psi, i):
    mass_term = -i * mass * psi
    return mass_term

def calc_spacial(gamma, dmu_psi, gterm, yterm):
    spatial = sum(gamma @ dmu_psi) + gterm + yterm
    return spatial

def calc_dirac(spatial, mass_term, gamma0_inv):
    """
    Kinetic derivation & coupling term of neighbor gauges
    """
    dirac = -gamma0_inv @ (spatial + mass_term)
    return dirac

# --------------------
# psi update
# --------------------

def calc_psi(psi, dt, dirac):
    # Eq: psi + dt * dirac
    psi = psi + dt * dirac
    return psi


# --------------------
# gterm (Gauge Coupling)
# --------------------

def calc_single_mu_gterm(i, g, field_value, T, psi):
    """
    -i * g * A_mu * (T @ psi)
    """
    return -i * g * field_value * (T @ psi)


def calc_gterm(psi, i, field_value, g, T):
    """
    Vectorized neighbor coupling calculation.
    """
    # Sum of coupling terms across all gauge fields and neighbors
    return sum(calc_single_mu_gterm(i, g, field_value, T, psi))


# --------------------
# gauss
# --------------------

def calc_gauss(x, mu=0, sigma=5):
    # Eq: exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return exp(-((x - mu) ** 2) / (2 * sigma ** 2))


# --------------------
# quark_doublet
# --------------------

def calc_quark_component(neighbor_psi, ckm_val):
    # Eq: psi_neighbor * V_ckm
    return neighbor_psi * ckm_val


def calc_quark_doublet(psi, total_down_psi_sum):
    # Eq: stack([psi, sum(neighbor_psi * ckm)])
    return stack([psi, total_down_psi_sum])
# --------------------
# Higgs Potential & Parameters
# --------------------
from jax.numpy import stack, sqrt


def calc_lambda_H(_mass, vev):
    # λ_H = m^2 / (2 v^2)
    return (_mass ** 2) / (2.0 * vev ** 2)

def calc_mu_higgs(vev, lambda_H):
    # μ = v * sqrt(λ)
    return vev * sqrt(lambda_H)

def calc_dV_dh(vev, lambda_H, h, mu):
    """
    Higgs potential derivative dV/dh.
    Note: 'mu' is passed from calc_mu_higgs.
    """
    # dV_dh = -mu^2 * (vev + h) + lambda_H * (vev + h)^3
    dV_dh = -mu ** 2 * (vev + h) + lambda_H * (vev + h) ** 3
    return dV_dh

# --------------------
# Laplacian & Derivatives
# --------------------

def calc_spatial_diff(field_forward, field_backward, d_space):
    """Zentraldifferenz: (Psi_{i+1} - Psi_{i-1}) / (2.0 * d)"""
    spacial_diffusion = (field_forward - field_backward) / (2.0 * d_space)
    return spacial_diffusion

def calc_time_diff(field_current, field_prev, d_time):
    """Zeitliche Ableitung: (Psi(t) - Psi(t-dt)) / dt"""
    t_diff = (field_current - field_prev) / d_time
    return t_diff

def calc_laplacian_h(h, d):
    """
    Sum of spatial derivatives for the Laplacian.
    In the pathway, roll/shift is treated as an operator.
    """
    # result = sum(calc_spatial_diff(roll(h, p), roll(h, m), d))
    return sum(calc_spatial_diff(h, h, d))

def calc_dmu_h(h, prev_h, d, dt):
    """
    Derivative vector for Higgs field.
    """
    time_res = calc_time_diff(h, prev_h, dt)
    spatial_res = calc_spatial_diff(h, h, d)
    return [time_res, spatial_res]


# --------------------
# Field Updates (Evolution)
# --------------------

def calc_mass_term_h(h, mass):
    return -(mass ** 2) * h

def calc_h_update(h, prev_h, dt, laplacian_h, mass_term, dV_dh):
    """
    Second-order time evolution for scalar field h.
    """
    dt2 = dt ** 2
    # h_new = 2.0 * h - prev_h + dt2 * (laplacian_h + mass_term - dV_dh)
    return 2.0 * h - prev_h + dt2 * (laplacian_h + mass_term - dV_dh)

def calc_phi(h, vev):
    """
    Returns the scalar doublet components.
    """
    # phi = [0.0, (vev + h) / sqrt(2)]
    return stack([0.0, (vev + h) / sqrt(2.0)])


# --------------------
# Energy Analysis
# --------------------

def calc_higgs_potential_energy(_mass, h, vev, lambda_h_val):
    """
    Potential energy density.
    Note: uses lambda_h_val as coupling factor.
    """
    m2 = _mass ** 2
    # 0.5 * m^2 * h^2 + λ * vev * h^3 + 0.25 * λ * h^4
    return 0.5 * m2 * h ** 2 + lambda_h_val * vev * h ** 3 + 0.25 * lambda_h_val * h ** 4

def calc_energy_density(dmu_h, potential_energy):
    """
    Total energy density: Kinetic + Gradient + Potential
    """
    kinetic = 0.5 * (dmu_h[0] ** 2)
    gradient = 0.5 * sum(dmu_h[1:] ** 2)
    return kinetic + gradient + potential_energy
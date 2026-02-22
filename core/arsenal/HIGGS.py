# -----------------------------------------------------------------------------
# HIGGS FIELD DYNAMICS - ATOMIC EQUATIONS
# -----------------------------------------------------------------------------

def calc_lambda_H(_mass, vev):
    # Self-coupling constant: λ_H = m^2 / (2 v^2)
    lambda_H = (_mass ** 2) / (2.0 * vev ** 2)
    return lambda_H

def calc_mu_sq(vev, lambda_H):
    # Squared mass parameter of the potential: μ^2 = v^2 * λ_H
    mu_sq = (vev ** 2) * lambda_H
    return mu_sq

def calc_dV_dh(vev, lambda_H, h, mu_sq):
    # Higgs potential derivative: dV/dh = -μ^2(v+h) + λ(v+h)^3
    dV_dh = -mu_sq * (vev + h) + lambda_H * (vev + h) ** 3
    return dV_dh

def calc_spatial_diff(field_forward, field_backward, d):
    # Central difference for spatial derivative: (ψ_p - ψ_m) / 2d
    spatial_diff = (field_forward - field_backward) / (2.0 * d)
    return spatial_diff

def calc_time_diff(field_current, field_prev, dt):
    # First order backward difference for time: (ψ_t - ψ_t-dt) / dt
    time_diff = (field_current - field_prev) / dt
    return time_diff

def calc_laplacian_h(spatial_diff_sum):
    # Summation of second derivatives or spatial differences for scalar laplacian
    laplacian_h = sum(spatial_diff_sum)
    return laplacian_h

def calc_mass_term(_mass, h):
    # Mass contribution to the Klein-Gordon equation: -m^2 * h
    mass_term = -(_mass ** 2) * h
    return mass_term

def calc_h(h, prev_h, dt, laplacian_h, mass_term, dV_dh):
    # Discrete second-order time evolution: h_new = 2h - h_prev + dt^2 * (∇^2 h - m^2h - dV/dh)
    h = 2.0 * h - prev_h + (dt ** 2) * (laplacian_h + mass_term - dV_dh)
    return h

def calc_phi_component(vev, h):
    # Physical Higgs component of the doublet: (v + h) / sqrt(2)
    phi_component = (vev + h) / (2.0 ** 0.5)
    return phi_component

def calc_kinetic_energy(time_diff):
    # Local kinetic energy density: 0.5 * (∂t h)^2
    kinetic_energy = 0.5 * (time_diff ** 2)
    return kinetic_energy

def calc_gradient_energy(spatial_diff_vector):
    # Local gradient energy density: 0.5 * |∇h|^2
    gradient_energy = 0.5 * sum(spatial_diff_vector ** 2)
    return gradient_energy

def calc_potential_energy(_mass, h, vev, lambda_H):
    # Higgs potential energy density: 0.5*m^2*h^2 + λ*v*h^3 + 0.25*λ*h^4
    potential_energy = 0.5 * (_mass ** 2) * (h ** 2) + lambda_H * vev * (h ** 3) + 0.25 * lambda_H * (h ** 4)
    return potential_energy

def calc_energy_density(kinetic_energy, gradient_energy, potential_energy):
    # Total energy density of the scalar field
    energy_density = kinetic_energy + gradient_energy + potential_energy
    return energy_density
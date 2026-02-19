# --------------------
# gg_coupling (Gluon-Gluon)
# --------------------

def calc_gg_eff_coupling(g_, g):
    # symmetric effective coupling strength
    return 0.5 * (g_ + g)

def calc_gg_interaction(g_eff, field_value, field_value_):
    # simplified antisymmetric field interaction term: [Aμ, Aν]
    return g_eff * (field_value * field_value_ - field_value_ * field_value)

def calc_gg_coupling(g_, field_value_, g, field_value):
    g_eff = calc_gg_eff_coupling(g_, g)
    return calc_gg_interaction(g_eff, field_value, field_value_)


# --------------------
# gf_coupling (Gluon-Fermion)
# --------------------

def calc_spinor_scalar(psi_bar, gamma_mu, psi_b):
    # psi_bar @ gamma_mu @ psi_b → Skalar
    return psi_bar @ gamma_mu @ psi_b

def calc_j_nu_gluon_quark(psi, psi_bar, gamma_mu, o_operator, g, quark_index):
    # Logic for b in range(3) summation
    # j_nu += (psi_bar @ gamma_mu @ psi[b]) * o_operator[quark_index, b]
    # Note: In the pathway, this sum is treated as an iterative or vectorized sum
    return g * sum(calc_spinor_scalar(psi_bar, gamma_mu, psi) * o_operator)

def calc_j_nu_non_qg(psi, psi_bar, gamma_mu, o_operator, g):
    # j_nu = g * (psi_bar @ gamma_mu @ o_operator @ psi)
    return g * (psi_bar @ gamma_mu @ o_operator @ psi)

def calc_gf_coupling(psi, psi_bar, field_value, g, gamma, o_operator):
    """
    Calculates Gluon-Fermion coupling.
    The logic decides between gluon_quark or non_qg based on context.
    """
    # Result is essentially the sum over indices: sum(j_nu * field_value[index])
    return sum(calc_j_nu_non_qg(psi, psi_bar, gamma, o_operator, g) * field_value)


# --------------------
# j_nu (Current Density)
# --------------------

def calc_j_nu(gg_coupling, gf_coupling):
    # j_nu = sum(gf_coupling) + sum(gg_coupling)
    return sum(gf_coupling) + sum(gg_coupling)


# --------------------
# field_value update
# --------------------

def calc_field_value(dt, dmu_fmunu, j_nu, field_value):
    """
    ∂_μ F^{μν} = j^ν -> Evolution step
    """
    # field_value = field_value + dt * (dmu_fmunu - j_nu)
    return field_value + dt * (dmu_fmunu - j_nu)


# --------------------
# fmunu (Field Strength Tensor)
# --------------------

def calc_fmunu(dmuG):
    """
    Calculates the antisymmetric tensor F from the derivative tensor dmu.
    F[mu, nu] = dmu[mu, nu] - dmu[nu, mu]
    """
    # term1 = dmu[:13, :4], term2 = dmu[:4, :13].T
    return dmuG - dmuG.T


# --------------------
# Derivatives (dmuG & dmu_fmunu)
# --------------------

def calc_spatial_diff(field_forward, field_backward, d_space):
    """Berechnet (Psi_{i+1} - Psi_{i-1}) / (2.0 * d)"""
    return (field_forward - field_backward) / (2.0 * d_space)

def calc_time_diff(field_current, field_prev, d_time):
    """Berechnet (Psi(t) - Psi(t-dt)) / dt"""
    return (field_current - field_prev) / d_time

def calc_dmuG(field_value, prev_field_value, d, dt):
    """
    Extracts time and spatial derivatives.
    In the extractor, 'roll' is treated as a shift operator.
    """
    time_part = calc_time_diff(field_value, prev_field_value, dt)
    spatial_part = calc_spatial_diff(field_value, field_value, d)
    return [time_part, spatial_part]

def calc_dmu_fmunu(fmunu, prev_fmunu, d, dt):
    """
    Derivative of the field strength tensor.
    """
    time_part = calc_time_diff(fmunu, prev_fmunu, dt)
    spatial_part = calc_spatial_diff(fmunu, fmunu, d)
    return [time_part, spatial_part]
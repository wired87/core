# --- Yang-Mills / Gluon-Gluon Dynamics ---

def calc_g_eff(g_neighbor, g_self):
    """Symmetric effective coupling strength."""
    g_eff = 0.5 * (g_neighbor + g_self)
    return g_eff

def calc_j_pair(g_eff, field_val_self, field_val_neighbor):
    """Antisymmetric field interaction term: [Aμ, Aν]."""
    j_pair = g_eff * (field_val_self * field_val_neighbor - field_val_neighbor * field_val_self)
    return j_pair

# --- Gluon-Fermion Interaction (Currents) ---

def calc_spinor_scalar(psi_bar, gamma_mu, psi_color):
    """Contraction of bar-spinor, gamma matrix and spinor: ψ_bar γ^μ ψ."""
    scalar = psi_bar @ gamma_mu @ psi_color
    return scalar

def calc_color_current(scalar, color_factor):
    """Weighting the spinor scalar with color matrix element."""
    c_current = scalar * color_factor
    return c_current

def calc_j_nu_base(g, psi_bar, gamma_mu, o_operator, psi):
    """Fundamental current density: g * (ψ_bar γ^μ T ψ)."""
    j_nu = g * (psi_bar @ gamma_mu @ o_operator @ psi)
    return j_nu

def calc_j_nu_mu(j_nu_base, field_mu):
    """Interaction current for specific component mu."""
    j_mu = j_nu_base * field_mu
    return j_mu

# --- Current Summation ---

def calc_j_total(gf_coupling, gg_coupling):
    """Total current density from fermion and boson contributions."""
    j_total = gf_coupling + gg_coupling
    return j_total

# --- Field Strength Tensor (F_munu) ---

def calc_f_munu(dmu_nu, dnu_mu):
    """Field strength tensor: F_μν = ∂_μ A_ν - ∂_ν A_μ."""
    f_munu = dmu_nu - dnu_mu
    return f_munu

# --- Derivatives (Finite Differences) ---

def calc_d_spatial(field_forward, field_backward, d_space):
    """Spatial central difference for field gradients."""
    d_grad = (field_forward - field_backward) / (2.0 * d_space)
    return d_grad

def calc_d_time(field_current, field_prev, d_time):
    """Temporal backward difference for field evolution."""
    d_time_val = (field_current - field_prev) / d_time
    return d_time_val

# --- Field Evolution (Proca / Maxwell Equation) ---

def calc_field_delta(dmu_fmunu, j_nu):
    """Source term for field update: (∂_μ F^μν - j^ν)."""
    f_delta = dmu_fmunu - j_nu
    return f_delta

def calc_field_update(field_value, dt, f_delta):
    """Time-stepping for the gauge field A_μ."""
    updated_field = field_value + dt * f_delta
    return updated_field
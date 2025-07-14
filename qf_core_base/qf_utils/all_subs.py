G_FIELDS=[
# Elektromagnetismus (U(1)_Y → nach Mischung: Photon)
            "photon",  # A_μ
            # Schwache Wechselwirkung (SU(2)_L)
            "w_plus",  # W⁺
            "w_minus",  # W⁻
            "z_boson",  # Z⁰
            # Starke Wechselwirkung (SU(3)_C)
            "gluon",  # G 8x4 array
]

FERMIONS=[
    # Leptonen
    "electron",  # ψₑ
    "myon",  # ψ_μ
    "tau",  # ψ_τ
    "electron_neutrino",  # νₑ
    "myon_neutrino",  # ν_μ
    "tau_neutrino",  # ν_τ

    # Quarks
    "up_quark",  # ψᵤ
    "down_quark",  # ψ_d
    "charm_quark",  # ψ_c
    "strange_quark",  # ψ_s
    "top_quark",  # ψ_t
    "bottom_quark"  # ψ_b
]

H = [
    "PHI"
]
ALL_SUBS=[*[f.upper() for f in FERMIONS],*[g.upper() for g in G_FIELDS],*H]

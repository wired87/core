G_FIELDS=[
# Elektromagnetismus (U(1)_Y → nach Mischung: Photon)
            "photon",  # A_μ
            # Schwache Wechselwirkung (SU(2)_L)
            "w_plus",  # W⁺
            "w_minus",  # W⁻
            "z_boson",  # Z⁰
            # Starke Wechselwirkung (SU(3)_C)
            #"gluon",  # G 8x4 array
*[f"gluon_{i}" for i in range(8)]

]


"""FERMIONS=[
    # Leptonen
    "electron",  # ψₑ
    "muon",  # ψ_μ
    "tau",  # ψ_τ
    "electron_neutrino",  # νₑ
    "muon_neutrino",  # ν_μ
    "tau_neutrino",  # ν_τ

    # Quarks
    "up_quark",  # ψᵤ
    "down_quark",  # ψ_d
    "charm_quark",  # ψ_c
    "strange_quark",  # ψ_s
    "top_quark",  # ψ_t
    "bottom_quark"  # ψ_b
]"""

#Err process: Too many indices for array: 2 non-None/Ellipsis indices for dim 1. [Array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
#jnp convert error: All input arrays must have the same shape. for item [[Array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],

H = [
    "higgs"
]



FERMIONS=[
    # Leptonen
    "electron",  # ψₑ
    "muon",  # ψ_μ
    "tau",  # ψ_τ
    "electron_neutrino",  # νₑ
    "muon_neutrino",  # ν_μ
    "tau_neutrino",  # ν_τ
    "up_quark_1",  # ψᵤ
    "up_quark_2",  # ψᵤ
    "up_quark_3",  # ψᵤ
    "down_quark_1",  # ψ_d
    "down_quark_2",  # ψ_d
    "down_quark_3",  # ψ_d
    "charm_quark_1",  # ψ_c
    "charm_quark_2",  # ψ_c
    "charm_quark_3",  # ψ_c
    "strange_quark_1",  # ψ_s
    "strange_quark_2",  # ψ_s
    "strange_quark_3",  # ψ_s
    "top_quark_1",  # ψ_t
    "top_quark_2",  # ψ_t
    "top_quark_3",  # ψ_t
    "bottom_quark_1"  # ψ_b
    "bottom_quark_2"  # ψ_b
    "bottom_quark_3"  # ψ_b
]


G_MAP=[
    "photon",  # A_μ
    # Schwache Wechselwirkung (SU(2)_L)
    "w_plus",  # W⁺
    "w_minus",  # W⁻
    "z_boson",  # Z⁰
    # Starke Wechselwirkung (SU(3)_C)
]

GLON_MAP = [f"gluon_{i}" for i in range(8)]

ALL_SUBS_LOWER=[*FERMIONS, *G_FIELDS, *H]
ALL_SUBS=[*[f.upper() for f in FERMIONS],*[g.upper() for g in G_FIELDS],*[h.upper() for h in H]]




# später unbedingt dynamischen nx graphen (lexicon) für parameter rewiring = pathfinder
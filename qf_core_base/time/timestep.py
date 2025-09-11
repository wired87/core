import numpy as np

def step_field(
    psi: np.ndarray,
    H_free: np.ndarray,
    V_int: np.ndarray,
    g_coupling: float,
    density: float,
    phi: float,
    dt: float,
    hbar: float = 1.054_571_817e-34,
):
    """
    dψ/dτ = -(i/ħ) H_local ψ,   H_local = H_free + g * density * V_int
    dτ = sqrt(g00) * dt  (statische Metrik).  -> dψ/dt = -(i/ħ) * sqrt(g00) * H_local ψ

    Args:
      psi      : (N,) Zustandsvektor (komplex)
      H_free   : (N,N) freier Hamiltonoperator (hermitesch)
      V_int    : (N,N) Wechselwirkungsoperator (hermitesch)
      g_coupling: Kopplungsstärke (Einheiten passend zu H)
      density  : lokale (Teilchen-/Energie-)Dichte, skaliert die Wechselwirkung
      g00      : Zeit-Zeit-Metrikkomponente am Ort (flach: g00=1, Nähe Masse: g00<1)
      dt       : Koordinaten-Zeitschritt (s)
      hbar     : reduziertes Planck-h (SI)

    Returns:
      psi_next : Zustand nach dt (erster-Ordnung-Schritt)
      d_tau    : zugehöriger Eigenzeit-Schritt
    """
    g00 = g00_from_phi(phi)
    d_tau = np.sqrt(g00) * dt
    H_local = H_free + g_coupling * density * V_int
    alpha = d_tau / hbar
    psi_next = psi - 1j * alpha * (H_local @ psi)   # Euler 1. Ordnung
    return psi_next, d_tau

def g00_from_phi(phi, c=299_792_458.0):
    # statische, schwache Felder: g00 ≈ 1 + 2φ/c^2  (φ<0 nahe Masse)
    return 1.0 + 2.0 * phi / (c**2)

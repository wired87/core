import numpy as np

from qf_core_base.qf_utils.field_utils import FieldUtils
from qf_core_base.symmetry_goups.sym_base import SymBase

class SU2(
    SymBase,
    FieldUtils
):

    def __init__(self):
        super().__init__()


    def _j_nu(self, psi, psi_bar, g, charge, hypercharge=None, fermid=None, assb=True): #psi, hypercharge, psi_bar, g
       #print("psi, ", psi)
       #print("psi_bar, ", psi_bar)
       #print("g", g)
        if assb is False:
            j_nu = np.zeros((4, 3), dtype=complex)
            for mu in range(4):
                for i in range(3):
                   #print("self.gamma[mu].shape", self.gamma[mu].shape)
                   #print("self.su2_group_generators[i].shape", self.su2_group_generators[i].shape)
                    j_nu[mu, i] += g * (
                            psi_bar @
                            self.gamma[mu] @
                            self.su2_group_generators[i] @
                            psi
                    )
            return j_nu
        else:
            j_nu = np.zeros(4, dtype=complex)
            for mu in range(4):
                j_nu[mu] = g * charge * (psi_bar @ self.gamma[mu] @ psi).item()
            return j_nu


    def j_nu_higgs(self, phi, dmu_phi, g):
        """
        Berechnet den Higgs-Strom:
        j_nu[a, mu] = i g [phi† T^a D_mu phi - (D_mu phi)† T^a phi]

        Args:
            phi (np.ndarray): (2,) komplex, Higgs-Dublett
            dmu_phi (np.ndarray): (4, 2) komplex, kovariante Ableitung D_mu phi
            g (float): Kopplungskonstante
            T (list[np.ndarray]): Liste mit 3 Generatoren (2x2 komplex)

        Returns:
            j_nu (np.ndarray): (4, 3) komplex, Strom
        """

        j_nu = np.zeros((4, 3), dtype=complex)
        T = self.su2_group_generators
        for mu in range(4):
            d_phi_mu = dmu_phi[mu]
            for a in range(3):
                term1 = phi.conj().T @ (T[a] @ d_phi_mu)
                term2 = d_phi_mu.conj().T @ (T[a] @ phi)
                j_nu[mu, a] = 1j * g * (term1 - term2)
        return j_nu

    def f_mu_nu(self, attrs, field_key, **args):
        F_mu_nu = np.zeros((4, 4))
        for mu in range(4):
            for nu in range(4):
                F_mu_nu[mu, nu] = attrs[f"dmu_{field_key}"][mu, nu] - attrs[f"dmu_{field_key}"][nu, mu]

    def update_gauge_field(self, field_value, F_mu_nu, j_nu, f_abc, d, dmu_F_mu_nu, **attrs):
        """
        SU(2) Update-Regel:
        D_μ F^{μν,a} = j^{ν,a}
        """
        W_mu_new = np.zeros_like(field_value)

        # Für jedes SU(2)-Index a:
        for a in range(3):
            # Divergence (hier exemplarisch)
            commutator = np.zeros_like(dmu_F_mu_nu)
            for b in range(3):
                for c in range(3):
                    commutator += f_abc[a, b, c] * field_value[b] * F_mu_nu[c]
            # Update-Regel
            W_mu_new[a] = field_value[a] + d["t"] * (dmu_F_mu_nu + commutator - j_nu[a])

        return W_mu_new

    def compute_self_interaction(self, field_value, g, f_abc):
        """
        Berechnet die Selbstkopplung für SU(2).
        A_mu: shape (4, 3)  # 4 Raumzeitrichtungen, 3 SU(2)-Komponenten
        f_abc: Strukturkonstanten epsilon_ijk
        """
        interaction = np.zeros_like(field_value, dtype=complex)
        for mu in range(4):
            for nu in range(4):
                for a in range(3):
                    commutator = 0.0
                    for b in range(3):
                        for c in range(3):
                            commutator += f_abc[a, b, c] * field_value[mu, b] * field_value[nu, c]
                    interaction[mu, a] += g * commutator
        return interaction



    def coupling_term(self, g, field_value, psi, **attrs):
        d_psi=np.zeros_like(psi.copy(), dtype=complex)
        for mu in range(4):
            P_L = 0.5 * (np.eye(4) - self.gamma5)
            d_psi += self.i * g * self.gamma[mu] @ field_value[mu] @ (P_L @ psi)
        return d_psi









    def higgs_g_coupling_term(self, g, field_value, phi, dmu_phi):
        # d_phi = Ableitung des Higgs-Feldes (4 Komponenten)
        dmu_phi_new = np.zeros_like(phi, dtype=complex)
        for mu in range(4):
            dmu_phi_new[mu] = dmu_phi[mu] + self.i * g * field_value[mu] @ phi
        return dmu_phi_new

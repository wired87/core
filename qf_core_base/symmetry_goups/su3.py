import numpy as np

from qf_core_base.symmetry_goups.sym_base import SymBase


class SU3(SymBase):

    def __init__(self):
        super().__init__()


    def _j_nu(self, psi, psi_bar, g, fermid, **attrs):
        j_nu = np.zeros((4, 8), dtype=complex)
        for mu in range(4):
            for a in range(8):
                j_nu[mu, a] += g * (psi_bar @ self.gamma[mu] @ self.su3_group_generators[a] @ psi) #.item()

        j_nu = np.zeros((4, 8), dtype=complex)
        return j_nu



    def update_gauge_field(self, field_value, F_mu_nu, j_nu, f_abc, dmu_F_mu_nu, d, **attrs):
        """
        SU(3) Update-Regel:
        D_μ F^{μν,a} = j^{ν,a}
        """
        G_mu_new = np.zeros_like(field_value)

        # Für jedes SU(3)-Index a:
        for a in range(8):
            commutator = np.zeros_like(dmu_F_mu_nu)
            for b in range(8):
                for c in range(8):
                    commutator += f_abc[a, b, c] * field_value[b] * F_mu_nu[c]

            G_mu_new[a] = field_value[a] + d["t"] * (dmu_F_mu_nu + commutator - j_nu[a])

        return G_mu_new

    def compute_self_interaction(self, field_value, g, f_abc):
        """
        Berechnet die Selbstkopplung für SU(3).
        A_mu: shape (4, 8)  # 4 Raumzeitrichtungen, 8 Gluon-Komponenten
        f_abc: SU(3)-Strukturkonstanten
        """
        interaction = np.zeros_like(field_value, dtype=complex)

        for mu in range(4):
            for nu in range(4):
                for a in range(8):
                    commutator = 0.0
                    for b in range(8):
                        for c in range(8):
                            commutator += f_abc[a, b, c] * field_value[mu, b] * field_value[nu, c]
                    interaction[mu, a] += g * commutator
        return interaction


    def coupling_term(self, g, field_value, psi):
        d_psi=np.zeros_like(psi.copy(), dtype=complex)
        for a in range(8):  # SU(3) Generatoren
            for mu in range(4):
                d_psi += self.i * g * self.gamma[mu] @ (field_value[a][mu] * (self.su3_group_generators[0][a] @ psi))
        return d_psi

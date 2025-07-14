import numpy as np

from qf_sim.physics.quantum_fields.qf_core_base.qf_utils.field_utils import FieldUtils


class Gluon(FieldUtils):
    def __init__(self):
        super().__init__()


    def f_mu_nu(self, g, field_value, d_field_value, f_abc):
        """
        Field strength tensor
        """
        f_mu_nu=np.zeros((4, 4, 8))
        for mu in range(4):
            for nu in range(4):
                for a in range(8):
                    deriv = d_field_value[mu, nu, a] - d_field_value[nu, mu, a]

                    # Selbstkopplungsteil
                    self_coupling = 0.0

                    for b in range(8):
                        for c in range(8):
                            self_coupling += f_abc[a, b, c] * field_value[mu, b] * field_value[
                                nu, c]

                    f_mu_nu[mu, nu, a] = deriv + g * self_coupling
        return f_mu_nu


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
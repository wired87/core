import numpy as np

from qf_core_base.symmetry_goups.sym_base import SymBase


class U1(SymBase):

    def __init__(self):
        super().__init__()

    def _j_nu(self, psi, hypercharge, psi_bar, g, fermid):
        # LOGGER.info("fermid", fermid)
        # LOGGER.info("psi type", type(psi), psi)
        # LOGGER.info("psi_bar type", type(psi_bar), psi_bar)
        # LOGGER.info("gamma type", type(self.gamma[0]), self.gamma[0])
        # LOGGER.info("hypercharge type", type(hypercharge), hypercharge)
        j_nu = np.zeros(4, dtype=complex)

        if "quark" in fermid:
            for i in range(3):
                for mu in range(4):
                    j_nu[mu] += g * hypercharge * (psi_bar[i] @ self.gamma[mu] @ psi[i])
        else:
            for mu in range(4):
                j_nu[mu] += g * hypercharge * (psi_bar @ self.gamma[mu] @ psi)
        return j_nu


    def j_nu_higgs(self, phi, d_phi, g):
        """
        Berechnet den Higgs-Strom für U(1):
        j_nu = i g [phi† ∂_nu phi - (∂_nu phi)† phi]

        Args:
            h (np.ndarray): (2,) komplex, Higgs-Dublett
            dmu_phi (np.ndarray): (4,2) komplex, Ableitungen ∂_nu phi
            g (float): Kopplungskonstante

        Returns:
            j_nu (np.ndarray): (4,) komplex
        """

        # LOGGER.info("phi",phi)
        # LOGGER.info("dmu_phi",d_phi, )
        # LOGGER.info("g",g)
        j_nu = np.zeros(4, dtype=complex)

        for mu in range(4):
            d_phi_mu = d_phi[mu]
            term1 = phi.conj().T @ d_phi_mu
            term2 = d_phi_mu.conj().T @ phi
            j_nu[mu] = 1j * g * (term1 - term2)

        return j_nu


    def f_mu_nu(self, attrs, field_key, **args):
       #print("attrs[u_{field_key}]", attrs[f"dmu_{field_key}"])
        F_mu_nu = np.zeros((4, 4), dtype=complex)
        for mu in range(4):
            for nu in range(4):
                dmunu =attrs[f"dmu_{field_key}"][mu][nu].item()
                dnumu = attrs[f"dmu_{field_key}"][nu][mu].item()
               #print("Calc dmunu with", dmunu, "-", dnumu, )
                F_mu_nu[mu, nu] = dmunu - dnumu
        return F_mu_nu

    def update_gauge_field(self, d, field_value, dmu_F_mu_nu, j_nu):
        """
        U(1) Update-Regel:
        ∂_μ F^{μν} = j^ν
        """
        # Beispiel: einfache Euler-Integration
        # A_mu_new = A_mu + delta_t * (∂_μ F^{μν} - j^ν)

        # Hier würdest du divergences etc. berechnen
        field_v_new = field_value + d["t"] * (dmu_F_mu_nu - j_nu)
        return field_v_new

    def compute_self_interaction_u1(self):
        """
        U(1): keine Selbstkopplung vorhanden.
        """
        return 0.0

    def coupling_term(self, g, field_value, psi, **attrs):
        """
        QED (U(1)
        quarks need to split before into its 3 components then run
        """
        d_psi = np.zeros_like(psi.copy(), dtype=complex)
        for mu in range(4):
            d_psi += self.i * g * self.gamma[mu] @ field_value[mu] @ psi
        return d_psi

    def higgs_g_coupling_term(self, field_value, phi, g, d_phi):
        """
        Kopplungsterm Gauge (U(1)) -> Higgs
        Für jedes µ:
        term[µ] = i * g * A_mu[µ] * phi

        Args:
            A_mu (np.ndarray): (4,) komplex, U(1)-Feld
            phi (np.ndarray): (2,) komplex, Higgs-Dublett
            g (float): Kopplungskonstante

        Returns:
            terms (np.ndarray): (4,2) komplex
        """
       #print("g", g)
       #print("field_value", field_value)
       #print("phi", phi)
        field_value = np.asarray(field_value).flatten()
        phi = np.asarray(phi).flatten()
        terms = np.zeros((4, 2), dtype=complex)
        for mu in range(4):
            terms[mu] = 1j * g * field_value[mu] * phi
       #print(f"i g Aμ ϕ = {terms}")
        return terms

    def compute_self_interaction(self, **attrs):
        # no self interaction in U1
        return
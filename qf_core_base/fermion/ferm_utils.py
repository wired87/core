
import numpy as np

from qf_core_base.qf_utils.field_utils import FieldUtils
from utils._np.serialize_complex import serialize_complex


class FermGCoupler(FieldUtils):
    """
    Calcs couplings between fermions and X
    Couplings Ferm -> G
    """

    def __init__(self):
        # save constants and terms both in edges
        # Coupling Constants between F and G
        super().__init__()
        self.coupling_constants = {
            "photon": self.photon_coupling,
            "z_boson": self.z_boson_coupling,
            "w_boson": self.w_boson_coupling,
            "gluon": self.gluon_coupling,
            "higgs": self.higgs_yukawa_coupling,
        }
























    # zwischen fermion und gauge feeldern ist immer "g" als kupplungskonstante aktiv

    def photon_coupling(self, g: float, theta_W: float) -> float:
        # e = g * sin(theta_W)
        return g * np.sin(theta_W)

    def z_boson_coupling(self, g: float, theta_W: float) -> float:
        # g_Z = g / cos(theta_W)
        return g / np.cos(theta_W)

    def w_boson_coupling(self, g: float) -> float:
        # W-Boson verwendet direkt g
        return g

    def gluon_coupling(self, g_s: float) -> float:
        # QCD: Gluon-Feld verwendet direkte Kopplungskonstante
        return g_s

    def higgs_yukawa_coupling(self, mass_fermion: float, vev: float) -> float:
        # Yukawa: Y_f = m_f / v (z.B. v = 246 GeV)
        return mass_fermion / vev



class FermUtils(
    FermGCoupler,

):

    def __init__(self):
        super().__init__()

    def coupling_ferm_zboson(
            self,
            psi,
            g,  # kupplungskonstante für gauges
            theta_W,
            psi_bar,
            gauge  # feld
    ):
        """
        All ferms couple to zBoson

        :return:
        """

        return g / np.cos(theta_W) * np.sum(psi_bar * np.sum(self.gamma * np.sum(gauge * psi)))
    def get_psi_left_right(self, psi):
        #print("get_psi_left_right psi", psi)
        psi_l = 0.0
        psi_r = 0.0
        for i in range(4):
            if i in [0, 1]:
                psi_l += psi[i]
            else:
                psi_r += psi[i]
        return psi_l, psi_r

    def check_spin(self, psi, is_quark):
        """
        Bestimmt die Chiralität deines Spinors:
        Links
        Rechts
        Gemischt oder Null
        """
        if is_quark is True:
            # quark
            psi = psi[0]

        #print("psi", psi)
        psi_L, psi_R = self.get_psi_left_right(psi)
        #print("psi_L, psi_R", psi_L, psi_R)

        norm_L = np.linalg.norm(psi_L)
        norm_R = np.linalg.norm(psi_R)
        #print("norm_L, norm_R", norm_L, norm_R)

        if norm_R > 0 and norm_L == 0:
            return "right"
        elif norm_L < 0 and norm_R == 0:
            return "left"
        elif norm_R == 0 and norm_L == 0:
            return "zero"
        else:
            return "mixed"


    def _isospin(self, handedness, ntype):
        # g_A Tabelle
        GA_TABLE = {
            "electron_neutrino": +0.5,
            "tau_neutrino": +0.5,
            "myon_neutrino": +0.5,
            "electron": -0.5,
            "muon": -0.5,
            "tau": -0.5,
            "up_quark": +0.5,
            "down_quark": -0.5,
        }

        if handedness == "right":
            return 0.0

        if handedness == "left":
            if ntype not in GA_TABLE:
                raise ValueError(f"Unbekannter Fermion-Typ '{ntype}'.")
            return GA_TABLE[ntype]

        elif handedness == "zero":
            return -0







    def _psi_bar(self, psi, is_quark):
        """
        Berechnet bar{ψ} = ψ† γ⁰ für Leptonen (4x1) und Quarks (3x4)

        Args:
            psi: np.ndarray – Spinor (4,1) oder (3,4)
            gamma0: np.ndarray – Dirac gamma^0 Matrix (4x4)

        Returns:
            bar_psi: np.ndarray – konjugierter Spinor
        """

        if is_quark is True:
            psi_bar = np.zeros((3, 4), dtype=complex)
            # Quark: Zeilenweise ψ† @ γ⁰ → Ergebnis ist (3,4)
            for i in range(3):
                psi_bar_i = psi[i].conj().T @ self.gamma[0]
                psi_bar[i]=psi_bar_i
        else:
            # Lepton: ψ† (1x4) @ γ⁰ (4x4) → (1x4)
            psi_bar = psi.conj().T @ self.gamma[0]

        #print("ψ†", psi_bar, psi_bar.shape)
        return psi_bar

    def _is_quark(self, type):
        ntype = type.upper()
        return ntype in [k.upper() for k in self.quarks]

    def init_psi(self, ntype, serialize=False, stim=True):
        random_stim = 1 if stim else 0
        if ntype.lower() in self.leptons:
            psi = np.array([
                [0.0 + 0.0j],
                [random_stim + 0.0j],
                [0.0 + 0.0j],
                [random_stim + 0.0j]
            ], dtype=complex)

        elif ntype.lower() in self.quarks:
            # Beispiel: Dirac-Spinor für einen up-Quark (Farbraum + Spinor → 3x4 Matrix)
            psi = np.array([
                [1.0 + 0.0j, random_stim + 0.0j, random_stim + 0.0j, 0.0 + 0.0j],  # Rot
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, random_stim + 0.0j],  # Grün
                [random_stim + 0.0j, random_stim + 0.0j, 1.0 + 0.0j, random_stim + 0.0j],  # Blau
            ], dtype=complex)
        else:
            raise ValueError(f"Invalid type {ntype}")

        if serialize is True:
            ##print("psi before serialization", psi, psi.shape)
            psi = serialize_complex(com=psi)

        ##print("psi set", psi)
        return psi








    ############
    # COUPLINGS
    ############

    def _calc_coupling_term_G(self, psi, field_value, g, ntype, total_g_coupling, is_quark):
        if is_quark:
            for i in range(3):
                coupling_term = self.fermion_gauge_coupling(
                    psi,
                    field_value,
                    g,
                    ntype
                )
                ct_array = np.array(coupling_term)
                total_g_coupling += ct_array
        else:
            coupling_term = self.fermion_gauge_coupling(
                psi,
                field_value,
                g,
                ntype
            )
            ct_array = np.array(coupling_term)
            total_g_coupling += ct_array

        #print(f"# _yukawa_couping_process finished: {total_g_coupling}")
        return total_g_coupling




    def _coupling_term_single(self, g, field_value_item, psi, T, gamma_item):
        return self.i * g * gamma_item @ (field_value_item * (T @ psi))

    def fermion_gauge_coupling(self, psi, field_value, g, ntype, theta_w=28.7, Q=1.0):
        """
        Fermion-Gauge-Kopplung nach SSB.

        Args:
            psi: Spinor (ndarray)
            field_value:
                Photon/Z: shape (4,)
                Gluon: shape (4,8)
                W: shape (4,)
            gamma: List mit gamma-Matrizen
            g: Kopplungskonstante
            ntype: 'photon', 'Z', 'gluon', 'W+','W-'
            theta_w: Weinbergwinkel (nur Z)
            Q: elektrische Ladung (nur Photon)
            su3_generators: Liste der Gell-Mann-Matrizen (nur Gluon)

        Returns:
            term: Spinor (ndarray)
        """
        term = np.zeros_like(psi, dtype=complex)

        if ntype == "photon":
            T = Q * np.identity(len(psi))
            for mu in range(4):
                term += self._coupling_term_single(
                    g,
                    field_value_item=field_value[mu],
                    T=T,
                    psi=psi,
                    gamma_item=self.gamma[mu]
                )

        elif ntype == "z_boson":
            T3 = self.su2_group_generators[2]
            Y = 0.5 * np.identity(2, dtype=complex)
            T = np.cos(theta_w) * T3 - np.sin(theta_w) * Y
            for mu in range(4):
                term += self._coupling_term_single(
                    g,
                    field_value_item=field_value[mu],
                    T=T,
                    psi=psi,
                    gamma_item=self.gamma[mu]
                )

        elif ntype == "gluon":
            if self.su2_group_generators is None:
                raise ValueError("su3_generators müssen übergeben werden!")
            for mu in range(4):
                for a in range(8):
                    T = self.su3_group_generators[a]
                    term += self._coupling_term_single(
                        g,
                        field_value_item=field_value[mu, a],
                        T=T,
                        psi=psi,
                        gamma_item=self.gamma[mu]
                    )

        elif ntype == "w_plus":
            T = np.array([[0, 1], [0, 0]], dtype=complex)
            for mu in range(4):
                term += self._coupling_term_single(
                    g,
                    field_value_item=field_value[mu],
                    T=T,
                    psi=psi,
                    gamma_item=self.gamma[mu]
                )

        elif ntype == "w_minus":
            T = np.array([[0, 0], [1, 0]], dtype=complex)
            for mu in range(4):
                term += self._coupling_term_single(
                    g,
                    field_value_item=field_value[mu],
                    T=T,
                    psi=psi,
                    gamma_item=self.gamma[mu]
                )
        else:
            raise ValueError(f"Unbekanntes ntype: {ntype}")

        return term


"""
    def photon_coupling_term(self, g, A_mu, psi, d_psi, **attrs):
        for mu in range(4):
            d_psi += self.i * g * self.gamma[mu] @ A_mu[mu] @ psi

    def z_boson_coupling_term(self, g, theta_W, Z_mu, psi, d_psi, **attrs):
        gZ = self.z_boson_coupling(g, theta_W)
        # return gZ * np.sum(psi_bar * np.sum(self.gamma * np.sum(Z_mu * psi)))
        for mu in range(4):
            d_psi += self.i * gZ * self.gamma[mu] @ Z_mu[mu] @ psi

    def w_boson_coupling_term(self, g, W_mu, psi, d_psi, **attrs):
        # return g * np.sum(psi_bar * np.sum(self.gamma * np.sum(W_mu * psi)))
        for mu in range(4):
            P_L = 0.5 * (np.eye(4) - self.gamma5)
            d_psi += self.i * g * self.gamma[mu] @ W_mu[mu] @ (P_L @ psi)

    def gluon_coupling_term(self, g, G_mu_a, psi, d_psi, **attrs):
        # Achtung: T_a sind SU(3)-Generatoren (Gell-Mann)


"""
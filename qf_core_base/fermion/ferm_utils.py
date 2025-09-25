
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
        ##print("get_psi_left_right psi", psi)
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
        Bestimmt die Chiralität eines Spinors:
        Links
        Rechts
        Gemischt oder Null
        """
        if is_quark is True:
            # quark
            psi = psi[0]

        ##print("psi", psi)
        psi_L, psi_R = self.get_psi_left_right(psi)
        ##print("psi_L, psi_R", psi_L, psi_R)

        norm_L = np.linalg.norm(psi_L)
        norm_R = np.linalg.norm(psi_R)
        ##print("norm_L, norm_R", norm_L, norm_R)

        if norm_R > 0 and norm_L == 0:
            return "right"
        elif norm_L < 0 and norm_R == 0:
            return "left"
        elif norm_R == 0 and norm_L == 0:
            return "zero"
        else:
            return "mixed"


    def _isospin(self, handedness, ntype):
        ntype=ntype.lower()
        # g_A Tabelle
        GA_TABLE = {
            "electron_neutrino": +0.5,
            "tau_neutrino": +0.5,
            "muon_neutrino": +0.5,
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
        try:
            if is_quark is True:
                psi_bar = np.zeros((3, 4), dtype=complex)
                # Quark: Zeilenweise ψ† @ γ⁰ → Ergebnis ist (3,4)
                for i in range(3):
                    psi_bar_i = psi[i].conj().T @ self.gamma[0]
                    psi_bar[i]=psi_bar_i
            else:
                # Lepton: ψ† (1x4) @ γ⁰ (4x4) → (1x4)
                psi_bar = psi.conj().T @ self.gamma[0]

            ##print("ψ†", psi_bar, psi_bar.shape)
            return psi_bar
        except Exception as e:
            print(f"Error in _psi_bar: {e}, psi: {psi}, is_quark: {is_quark}")

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
            ###print("psi before serialization", psi, psi.shape)
            psi = serialize_complex(com=psi)

        ###print("psi set", psi)
        return psi


    ############
    # COUPLINGS
    ############


    def _calc_coupling_term_G(
            self,
            psi,
            field_value,
            g,
            ntype,
            gluon_index,
            total_g_coupling,
            is_quark,
            handedness):
        if is_quark is True:
            #print("Quark deteted")
            coupling_term = self.fermion_gauge_coupling(
                psi,  # without time
                field_value,
                g,
                ntype,
                gluon_index,
                handedness,
            )
            #print(f"coupling_term: {coupling_term}")
            #ct_array = np.array(coupling_term)
            #print("ct_array", coupling_term.shape)
            total_g_coupling += coupling_term
        else:
            #print("!Quark")
            coupling_term = self.fermion_gauge_coupling(
                psi,
                field_value,
                g,
                ntype,
                gluon_index,
                handedness,
            )
            #ct_array = np.array(coupling_term)
            #print(f"coupling_term: {coupling_term, coupling_term.shape}")
            total_g_coupling += coupling_term
        return total_g_coupling



    def get_gauge_generator(
            self,
            ntype,
            handedness,
            gluon_index=None,
            psi=None,
            Q=1.0,
            theta_w=28.7,

    ):
        ntype = ntype.lower()
        if ntype == "photon":
            T = Q * np.identity(len(psi))
            #print("p T", T)

        elif ntype == "z_boson":
            T3 = self.su2_group_generators[2]
            Y = 0.5 * np.identity(2, dtype=complex)
            T = np.cos(theta_w) * T3 - np.sin(theta_w) * Y

            #print(f"zT final: {T}")
        elif ntype == "gluon":
            T = self.su3_group_generators[gluon_index]
            #print("gluon T", T)

        elif ntype == "w_plus":
            T = np.array([[0, 1], [0, 0]], dtype=complex)
            #print("w+ T", T)

        elif ntype == "w_minus":
            T = np.array([[0, 0], [1, 0]], dtype=complex)
            #print("w- T", T)
        return T


    def extract_psi_lrm(self, psi, handedness, is_quark):
        #print(f"self.handedness: {handedness}")
        #print(f"psi: {psi, psi.shape}")
        if is_quark is False:
            if handedness == "left":
                """P_L = 0.5 * (np.eye(4) - self.gamma5)
                return P_L @ psi"""
                #psi_L = 0.5 * (np.eye(4) - self.gamma5) @ psi  # ergibt (4,1)
                return psi[:2]
            elif handedness == "right":
                """P_R = 0.5 * (np.eye(4) + self.gamma5)
                return P_R @ psi"""
                #psi_R = 0.5 * (np.eye(4) + self.gamma5) @ psi
                return psi[2:]
            elif handedness == "mixed":
                psi_R = 0.5 * (np.eye(4) + self.gamma5) @ psi
                psi_R = psi_R[2:]
                psi_L = 0.5 * (np.eye(4) - self.gamma5) @ psi  # ergibt (4,1)
                psi_L= psi_L[:2]
                #print("psi_L, psi_R", psi_L, psi_R)
                return psi_L, psi_R
        else:
            # singlet quark (3,4) coupling
            return psi



    def fermion_gauge_coupling(
            self,
            psi,
            field_value,
            g,
            ntype,
            gluon_index,
            handedness,
            index=None
    ):
        """
        Fermion-Gauge-Kopplung nach SSB.
        Ja. W⁺/W⁻ koppeln ausschließlich an linkshändige Fermionen.
        """

        term = np.zeros_like(psi, dtype=complex)
        T = self.get_gauge_generator(
            ntype,
            handedness,
            gluon_index,
            psi,
        )
        ##printer(locals())
        for mu in range(4):
            term += -self.i * g * field_value[mu] * (T @ psi)
        return term

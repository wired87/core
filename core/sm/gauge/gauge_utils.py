import itertools

from qf_utils.field_utils import FieldUtils

import numpy as np

class GaugeHiggsCoupling:

    def __init__(self):
        self.coupling_constants = {
            "w_boson": self.w_higgs_coupling,
            "z_boson": self.z_higgs_coupling,
        }

        self.coupling_terms = {
            ("higgs", "w_boson"): self.w_higgs_coupling_term,
            ("higgs", "z_boson"): self.z_higgs_coupling_term,
        }

    def w_higgs_coupling(self, g: float) -> float:
        # Standard W-Higgs-Kopplung: g / 2
        return g / 2

    def z_higgs_coupling(self, g: float, theta_W: float) -> float:
        # Z-Higgs-Kopplung: g / (2 * cos(theta_W))
        return g / (2 * np.cos(theta_W))

    #################
    ### After SSB ###

    def higgs_w_coupling_term(self, H, W_plus_mu, W_minus_mu, g, v):
        """
        L_HWW = g^2 * v * H * W^+_mu * W^-^mu
        """
        return g ** 2 * v * H * np.dot(W_plus_mu, W_minus_mu)

    def higgs_z_coupling_term(self, H, Z_mu, g, g_prime, v):
        """
        L_HZZ = 0.5 * (g^2 + g'^2) * v * H * Z_mu * Z^mu
        """
        return 0.5 * (g ** 2 + g_prime ** 2) * v * H * np.dot(Z_mu, Z_mu)

    def higgs_gluon_loop_term(self, H, G_mu_nu_a, alpha_s, v):
        """
        L_Hgg ~ (alpha_s / (12 * pi * v)) * H * G^a_{mu nu} * G^{a mu nu}
        """
        factor = alpha_s / (12 * np.pi * v)
        contraction = np.sum(G_mu_nu_a * G_mu_nu_a)  # Sum over a, mu, nu
        return factor * H * contraction


class GaugeUtils(FieldUtils):
    """
    G -> Ferms fmunu niht enthalten da Die Kopplung an würde z. B. magnetische Momente oder Anomalien beschreiben – aber das ist eine höhere Ordnung und nicht Teil des Basis-Lagrangians.
    fmuni in jedem gauge enthalten

    alles als Yang-Mills-Theorie
    """
    def __init__(self):
        super().__init__()
        self.fabc = self.get_fabc()

    def get_fabc(self):
        """
        Shape (8,8,8):
        Automatische Berechnung f^{abc} aus den Generatoren.
        Returns:
            f_abc: np.ndarray, shape (N,N,N)
            Strukturkonstante
        """
        T = self.o_operators
        N = len(T)
        f_abc = np.zeros((N, N, N), dtype=float)

        for a in range(N):
            for b in range(N):
                commutator = T[a] @ T[b] - T[b] @ T[a]
                for c in range(N):
                    # Tr([T^c][T^a,T^b]) = i f^{abc} * (Normfaktor)
                    trace_val = np.trace(T[c] @ commutator)
                    # Für SU(N) üblich normiert auf 2*i
                    f_abc[a, b, c] = np.real((1 / (2j)) * trace_val)

        return f_abc



    def _check_active_indizes(self, neighbors:list[tuple],nattrs=None):
        #
        # mark: gluons treated as single entries
        filtered_neighbors = []
        for nid, nattrs in neighbors:
            nntype = nattrs.get("type").lower()
            nfield_key = self._field_value(nntype)
            field_value = nattrs.get(nfield_key)
            non_zero = np.any(field_value != 0)
            if non_zero:
                filtered_neighbors.append(
                    (nid, nattrs)
                )

        #print("_check_active_indizes finieshed")
        return filtered_neighbors



    def get_O_operator(self, ntype, Q=None, g_V=None, g_A=None, su3_generators=None, gluon_index=None, gamma5=None):
        """
        Liefert den Operator O für das gegebene Eichfeld ntype.

        Args:
            ntype (str): Name des Feldes ("photon", "Z", "gluon", "W")
            Q (float): Elektrische Ladung (nur für Photon)
            g_V (float): Vektor-Kopplung (nur für Z)
            g_A (float): Axial-Kopplung (nur für Z)
            su3_generators (list[np.ndarray]): Liste der SU(3)-Generatoren (nur für Gluon)
            gluon_index (int): Index des Gluon-Generators (0..7)
            gamma5 (np.ndarray): Gamma5-Matrix (nur für Z)

        Returns:
            np.ndarray: Operator-Matrix (4x4)
        """
        if ntype == "photon":
            if Q is None:
                raise ValueError("Für 'photon' muss Q angegeben werden.")
            return Q * np.eye(4, dtype=complex)

        elif ntype == "z_boson":
            if g_V is None or g_A is None or gamma5 is None:
                raise ValueError("Für 'Z' müssen g_V, g_A und gamma5 angegeben werden.")

        elif ntype == "gluon":
            if su3_generators is None or gluon_index is None:
                raise ValueError("Für 'gluon' müssen su3_generators und gluon_index angegeben werden.")
            return su3_generators[gluon_index]

        elif ntype == "w_boson": # todo change
            # Für W± ist O einfach die Einheitsmatrix (Flavorwechsel ist extern)
            return np.eye(4, dtype=complex)

        else:
            raise ValueError(f"Unbekanntes ntype '{ntype}'")

    def j_nu_higgs(self, new_j_nu, phi, d_phi, g, ntype, theta_w):
        # higgs kuppelt nicht an GLUON also alle 4, shape
        # jnu higgs und jnu fermions werden in den spezisfischen gauge feldern summiert
        #printer(locals())
        T = self.define_generator(ntype, theta_w)
        #print("T", T)
        for mu in range(4):
            d_phi_mu = d_phi[mu]
            term1 = phi.conj().T @ (T @ d_phi_mu)
            term2 = d_phi_mu.conj().T @ (T @ phi)
            new_j_nu[mu] += self.i * g * (term1 - term2)
        return new_j_nu



    def define_generator(self, ntype, theta_w):
        ntype = ntype.lower()
        #nach ssb gelten immernoch urspr. generstoren
        T3 = self.T3
        Y = 0.5 * np.identity(2, dtype=complex)

        if ntype == "photon":
            T = np.sin(theta_w) * T3 + np.cos(theta_w) * Y
        elif ntype == "z_boson":
            T = np.cos(theta_w) * T3 - np.sin(theta_w) * Y
        elif ntype == "w_plus":
            # Ladder operator T+
            T = np.array([[0, 1], [0, 0]], dtype=complex)
        elif ntype == "w_minus":
            # Ladder operator T-
            T = np.array([[0, 0], [1, 0]], dtype=complex)
        else:
            T = None
        return T





    ##############
    # GG Couplings
    ############## TRIPPLES

    def j_nu_w_photon(self, W_plus, dW_plus, W_minus, dW_minus, e):
        """
        todo gleichungen werdne nodes
        Dreifach-Kopplung W–W–Photon als Stromquelle für Photon.
        Alle Inputs: shape (4,) komplex
        Returns: j_nu shape (4,)
        """
        import numpy as np
        j_nu = np.zeros(4, dtype=complex)
        for nu in range(4):
            sum1 = 0.0
            sum2 = 0.0
            for mu in range(4):
                F_plus = dW_plus[mu, nu]
                F_minus = dW_minus[mu, nu]
                sum1 += F_plus * W_minus[mu]
                sum2 += F_minus * W_plus[mu]
            j_nu[nu] = -1j * e * (sum1 - sum2)
        return j_nu



    def j_nu_w_z(self, w_plus, dw_plus, w_minus, dw_minus, g, theta_w):
        """
        WWZ-Kopplung nach SSB, Strom für das Z-Feld.
        """
        import numpy as np
        j_nu = np.zeros(4, dtype=complex)
        cos_theta = np.cos(theta_w)

        for nu in range(4):
            sum1 = 0.0
            sum2 = 0.0
            for mu in range(4):
                F_minus = dw_minus[nu, mu]
                F_plus = dw_plus[nu, mu]
                # Beachte Reihenfolge: W^{nu mu} = d^nu W^mu - d^mu W^nu
                sum1 += F_minus * w_plus[mu]
                sum2 += F_plus * w_minus[mu]
            j_nu[nu] = 1j * g * cos_theta * (sum1 - sum2)
        return j_nu


    def j_nu_ww_aa(self, W_plus, W_minus, A_field, e):
        """
        Vierfach-Kopplung W–W–Photon–Photon als vereinfachter Term.
        """
       #print("=== Inputs ===")
       #print("W_plus:")
       #print(W_plus)
       #print("W_minus:")
       #print(W_minus)
       #print("A_field:")
       #print(A_field)
       #print("e:")
       #print(e)
       #print("=================")

        import numpy as np
        j_nu = np.zeros(4, dtype=complex)
        for nu in range(4):
            term1 = 0.0
            term2 = 0.0
            for mu in range(4):
                term1 += W_plus[mu] * W_minus[mu] * A_field[nu]
                term2 += W_plus[mu] * A_field[mu] * W_minus[nu]
            j_nu[nu] = -0.5 * e ** 2 * (term1 - term2)
        return j_nu

    def _e(self, g):
        """
        Berechnet die elektromagnetische Kopplungskonstante e aus
        SU(2)-Kopplung g und Weinberg-Winkel theta_w (in Radiant).
        Auch nach ssb
        """
        import numpy as np
        e = g * np.sin(getattr(self, "theta_W"))
        return e

    def ceate_powerset(
            self,
            ntype,
            nid,
            valid_vertex:list,
            neighbors:dict,
            self_nattrs:dict,
            quad=True
    ):
        self_ntype=ntype.upper()
        clasified_neighbors = {}

        self_struct = {"id": nid, **self.restore_selfdict(self_nattrs)}

        if "gluon" not in valid_vertex:
            for i, vtype in enumerate(valid_vertex):
                vtype = vtype.upper()
                #print("neighbors")
                #pprint.pp(neighbors)
                #print("self attrs")
                #pprint.pp(self_nattrs)
                if vtype != self_ntype:
                    for nid, nattrs in neighbors[vtype.upper()].items():
                        print("nid", nid)
                        print("nattrs", nattrs)
                        neighbor_struct = {"id": nid, **self.restore_selfdict(nattrs)}

                        # Create empty space
                        if vtype.upper() not in clasified_neighbors:
                            clasified_neighbors[vtype.upper()] = []

                        if vtype.upper() != self_ntype.upper() or (
                            vtype.upper() == "GLUON" and i != 2) or (
                            vtype.upper() != self_ntype.upper()
                        ):  # -> or check to leave last item free (for self)
                            clasified_neighbors[vtype.upper()].append(
                                neighbor_struct  # single keys
                            )

            powerset = list(itertools.product(
                *clasified_neighbors.values()
            ))

            print("powersetA:")

        else:
            # create
            print("len gluon:", len(neighbors["GLUON"]))
            powerset = list(
                itertools.combinations(
                    [
                        {"id": nid, **self.restore_selfdict(nattrs)}
                        for nid, nattrs in neighbors["GLUON"].items()
                    ],
                    r=3 if quad is True else 2
                )
            )
            print("len powerset", len(powerset))
            ##pprint.pp(powerset)

        full_powerset = []
        # Add self struct to powerset
        for variation in powerset:
            #print("variation", variation)
            full_powerset.append(variation + (self_struct,))
        print("full_powerset", len(full_powerset))

        return full_powerset


"""

def calculate_f_mu_nu_final_wrapper(self, attrs) -> np.ndarray:

    The outer function handling the static string conversion and exception,
    then calling the JAX-optimized core function.

    ntype_str = attrs.get('ntype', '').lower()

    # 1. Determine the non-Abelian switch (must be a static value)
    if ntype_str in ["w_plus", "w_minus"]:
        is_non_abelian_switch = 1.0
        # Initialize self_i if not present (assuming it comes from self in the original class)
        if 'self_i' not in attrs:
            attrs['self_i'] = np.complex64(0.0 + 1.0j)
    elif ntype_str in ["photon", "z_boson", "gluon"]:
        is_non_abelian_switch = 0.0
        if 'self_i' not in attrs:
            attrs['self_i'] = np.complex64(0.0 + 1.0j)
    else:
        # Cannot be avoided in Python for correctness if 'ntype' is unknown
        raise ValueError(f"Unbekanntes ntype: {ntype_str}")

    # 2. Call the JAX core function
    return _calculate_f_mu_nu(attrs, is_non_abelian_switch)

        if gluon_index is not None and nntype is not None and "quark" in nntype.lower():
            j_nu = self.j_nu_gluon_quark(psi, psi_bar, self.gamma[index], o_operator, g)
        else:
            j_nu = g * (psi_bar @ self.gamma[index] @ o_operator @ psi)  # take whole psi = correct
        return j_nu
        
        
        

    def photon_field_value(self, theta_W, W3_mu, B_mu):
        return np.sin(theta_W) * W3_mu + np.cos(theta_W) * B_mu

    def z_boson_field_value(self, theta_W, W3_mu, B_mu):
        return np.cos(theta_W) * W3_mu - np.sin(theta_W) * B_mu

    def w_plus_field_value(self, W1_mu, W2_mu):
        return 1 / np.sqrt(2) * (W1_mu - 1j * W2_mu)

    def w_minus_field_value(self, W1_mu, W2_mu):
        return 1 / np.sqrt(2) * (W1_mu + 1j * W2_mu)


"""

"""
def fmunu_wpm(self, field_value, d_field_value, g):
    F = np.zeros((4, 4, 2, 2), dtype=complex)
    for mu in range(4):
        for nu in range(4):
            comm = field_value[mu] @ field_value[nu] - field_value[nu] @ field_value[mu]
            # print("comm", comm)
            F[mu, nu] = d_field_value[mu][nu] - d_field_value[nu][mu] + self.i * g * comm
    return F
"""
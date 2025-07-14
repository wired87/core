import json
import pprint
from cmath import sqrt

import numpy as np
import inspect

from qf_sim.physics.quantum_fields.lexicon import QFLEXICON
from qf_sim.physics.quantum_fields.nodes.fermion import PSI_PSI_EQ, FERM_HIGGS_EQ, FERM_GAUGE_EQ, PSI_EQ
from qf_sim.physics.quantum_fields.nodes.g import GAUGE_EQ
from qf_sim.physics.quantum_fields.nodes.higgs import PHI_EQ, PHI_PHI_EQ, PHI_GAUGE_EQ
from utils._np.serialize_complex import deserialize_complex
from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER


class CalcUtils:
    """
    Definition einzelner code properties der Equation - nodes
    """

    def __init__(self, g: GUtils):
        self.g = g
        self.arsenal = {
            "psi": [
                *FERM_HIGGS_EQ,
                *PSI_PSI_EQ,
                *FERM_GAUGE_EQ,
                *PSI_EQ,
            ],
            "phi": [
                *PHI_EQ,
                *PHI_GAUGE_EQ,
                *PHI_PHI_EQ,
            ],
            "gauge": [
                *GAUGE_EQ
            ]
        }

    def mu_squared_H(self, lambda_H=None, vev=None):
        return -lambda_H * vev ** 2

    def phi_plus(self, phi=None):
        return phi[0] + 1j * phi[1]

    def phi_minus(self, phi=None):
        return phi[0] - 1j * phi[1]

    def phi_zero(self, phi=None):
        return phi[2] + 1j * phi[3]

    def compute_vev(self, lambda_H=None, mu_squared_H=None):
        return sqrt(-mu_squared_H / lambda_H)

    def dmu_phi(self, d=None, nphi=None):
        return np.sum([(p - m) / (2 * d[k]) for k, (p, m) in nphi.items()])

    def laplacian_H(self, dx=None, nlen=None, nphi=None, phi=None):
        return (sum(nphi) - nlen * phi) / dx ** 2

    def potential_energy(self, lambda_H=None, mu_squared_H=None, phi=None, vev=None):
        return mu_squared_H * abs(phi) ** 2 + lambda_H * abs(phi) ** 4

    def higgs_kinetic_term(self, dmu_phi=None):
        return 0.5 * dmu_phi * dmu_phi

    def potential_energy(self, lambda_H=None, phi=None, vev=None):
        return lambda_H * (phi ** 2 - vev ** 2) ** 2

    def higgs_total_energy(self, higgs_kinetic_term=None, potential_energy_H=None):
        return potential_energy_H + higgs_kinetic_term

    def higgs_z_coupling(self, M_Z=None, Z_mu=None, g=None, phi=None, theta_W=None):
        return 0.5 * g / np.cos(theta_W) * M_Z * phi * np.dot(Z_mu, Z_mu)

    def higgs_z_coupling(self, Z_mu=None, g=None, phi=None, theta_w=None):
        return (g / np.cos(theta_w)) ** 2 * np.sum(np.abs(phi) ** 2 * Z_mu ** 2)

    def higgs_w_coupling(self, M_W=None, W_mu_minus=None, W_mu_plus=None, g=None, phi=None):
        return g * M_W * phi * np.dot(W_mu_plus, W_mu_minus)

    def higgs_w_coupling(self, W_mu=None, g=None, phi=None):
        return g ** 2 * np.sum(np.abs(phi) ** 2 * W_mu ** 2)

    def higgs_potential_force(self, lambda_H=None, phi=None, vev=None):
        return 4 * lambda_H * phi * (phi ** 2 - vev ** 2)

    def calculate_field_strength_tensor(self, d_mu_A_nu=None, d_nu_A_mu=None):
        return d_mu_A_nu - d_nu_A_mu

    def gluon_field_strength(self, gluon_field_tensor=None):
        return -0.25 * np.sum(gluon_field_tensor ** 2)

    def em_kinetic_term(self, dA_mu=None, dA_nu=None):
        return -0.25 * (dA_mu - dA_nu) ** 2

    def compute_gluon_kinetic_term(self, gluon_field_tensor=None):
        return -0.25 * np.einsum('μν,μν', gluon_field_tensor, gluon_field_tensor)

    def compute_em_kinetic_term(self, F_mu_nu=None):
        return -0.25 * np.einsum('μν,μν', F_mu_nu, F_mu_nu)

    def compute_weak_kinetic_term(self, weak_field_strength=None):
        return -0.25 * np.einsum('μν,μν', weak_field_strength, weak_field_strength)

    def local_fermion_energy(self, dmu_psi=None, m_psi=None, psi=None, psi_bar=None):
        return 0.5 * np.dot(dmu_psi, dmu_psi) + m_psi * np.dot(psi_bar, psi)

    def dirac_kinetic_term(self, dmu_psi=None, gamma_mu=None, i=None, psi_bar=None):
        return i * np.sum(np.dot(psi_bar, np.dot(gamma_mu, dmu_psi)))

    def psi_bar_definition(self, gamma=None, psi=None):
        return np.dot(gamma[0], psi).conj().T

    def set_neighors_plus_minus(self, node_id, self_attrs, d: int, trgt_type="QFN", field_type="phi"):
        """
        :return: qfn neighbors foreach pos dir (p + m)
        (x before and x after)
        """
        LOGGER.info("Set neighbors pm")
        direction_definitions = {
            "x": (1, 0, 0),
            "y": (0, 1, 0),
            "z": (0, 0, 1),

        }

        nsum_phi = {}
        self_pos = np.array(self_attrs["pos"])

        all_nodes = [(k, v) for k, v in self.g.G.nodes(data=True) if v.get("type") == trgt_type]
        node_pos_dict = {node: np.array(attrs.get("pos")) for node, attrs in all_nodes}

        for direction_name, direction_matrix in direction_definitions.items():
            offset = np.array(direction_matrix) * d
            pos_plus = self_pos + offset
            pos_minus = self_pos - offset

            node_plus = next((k for k, v in node_pos_dict.items() if np.allclose(v, pos_plus)), None)
            node_minus = next((k for k, v in node_pos_dict.items() if np.allclose(v, pos_minus)), None)

            # Fallback auf self_node, falls Ziel nicht gefunden
            node_plus = node_plus if node_plus else node_id
            node_minus = node_minus if node_minus else node_id

            np_attrs = self.g.G.nodes[node_plus]
            nm_attrs = self.g.G.nodes[node_minus]

            nsum_phi.update(
                {
                    direction_name: [
                        np_attrs.get("id"),  # + npsi
                        nm_attrs.get("id"),  # - npsi
                    ]
                }
            )

        return self_attrs

    def _dmuX(
            self,
            field_forward,
            field_backward,
            d,
            self_time=None,
            self_time_prev=None,
            plus_time=None,
            minus_time=None,

            time=False
    ):
        #print("field_forward", field_forward)
        #print("field_backward", field_backward)
        if time is False:
            # convert neighbor phi to complex
            ##print("phi_forward, phi_backward", phi_forward, phi_backward)
            field_forward = np.array(field_forward, dtype=complex)
            field_backward = np.array(field_backward, dtype=complex)

            # phi_forward=self._convert_to_complex(phi_forward)
            # phi_backward=self._convert_to_complex(phi_backward)
            """delta_t_plus = plus_time - self_time
            delta_t_minus = self_time - minus_time
            d = delta_t_plus + delta_t_minus"""
        else:
            #d = 2
            pass

        single_dphi = (field_forward - field_backward) / d
        return single_dphi






    def d_psi_quarters(self, d_X: np.ndarray) -> np.ndarray:
        """
        Nimmt ein 1D-Array mit 26 Nachbarwerten und teilt es in 4 gleich große
        Teile (je 6 Werte + verbleibend 2).
        Summiert die Werte in jedem Teil und gibt vier Summen zurück.

        Rückgabe:
            np.ndarray mit shape (4,) mit den Summen der vier Gruppen.
        """
        assert d_X.ndim == 1 and d_X.size == 26, \
            "psi_neighbors muss 1D mit Länge 26 sein"

        # in 4 fast-gleiche Teile teilen
        # wir bekommen z. B. [6,6,7,7]
        indices = np.array_split(np.arange(26), 4)




        sums = np.zeros(4, dtype=d_X.dtype)
        for i, idx in enumerate(indices):
            sums[i] = np.sum(d_X[idx])
        return sums



    def _dmu(self, self_ntype, attrs, d, neighbors_pm, field_key="h"):
        # todo auf 26 nachbar updaten
        # update nachbarm
        # adde merh params

        #print("neighbors_pm", neighbors_pm)

        phi_t = self._dmuX(
            attrs[field_key],
            attrs[f"prev_{field_key}"],
            d=d["t"],  # dt = timestep todo improve (no global time
            time=True
        )

        dmu = [
            phi_t
        ]

        ##print("neighbors", self.neighbors_pm)
        for i, (key, pm) in enumerate(neighbors_pm.items()):  # x,y,z
            #print("Dphi run", i)
            plus_id = pm[0]
            minus_id = pm[1]

            neighhbor_plus = self.g.get_single_neighbor_nx(plus_id, self_ntype.upper())
            neighhbor_minus = self.g.get_single_neighbor_nx(minus_id, self_ntype.upper())


            # Unpack serialized field values
            plus_value_raw = neighhbor_plus[1][field_key]
            minus_value_raw = neighhbor_minus[1][field_key]

            # Deserialize field content
            phi_plus = deserialize_complex(bytes_struct=plus_value_raw)
            phi_minus = deserialize_complex(bytes_struct=minus_value_raw)


            #print(f"nminus {plus_id}-> {neighhbor_minus[1]}:{phi_minus}")

            if phi_plus is None:
                phi_plus = attrs[field_key]
            if phi_minus is None:
                phi_minus = attrs[field_key]

            dmu_x = self._dmuX(phi_plus, phi_minus, d[key])

            #print(f">>>dmu{i}:", dmu_x)
            dmu.append(dmu_x)
        #print(f"Finished dmu forkey: {field_key}")
        #return self.d_psi_quarters(d_X=dmu)
        return dmu

    ################
    def _convert_eq2dict(self, method, name):
        for name, method in self.all_equations:
            return {
                "name": name,
                "code": json.dumps(method),
                "parameters": self._extract_method_meta(method)
            }

    def _extract_method_meta(self, method):
        # Signature-Objekt abrufen
        signature = inspect.signature(method)
        return [{
            "name": param.name,
            "type": param.annotation,
        }
            for name, param in signature.parameters.items()
        ]

"""
"xy_pp": (1, 1, 0),
"xy_pm": (1, -1, 0),
"xz_pp": (1, 0, 1),
"xz_pm": (1, 0, -1),
"yz_pp": (0, 1, 1),
"yz_pm": (0, 1, -1),
"xyz_ppp": (1, 1, 1),
"xyz_ppm": (1, 1, -1),
"xyz_pmp": (1, -1, 1),
"xyz_pmm": (1, -1, -1)



LOGGER.info(f"Calc dmu nsum: {nsum}")
LOGGER.info(f"Calc dmu prev: {prev}")
LOGGER.info(f"Calc dmu now: {now}")
LOGGER.info(f"Calc dmu d: {d}")
LOGGER.info(f"Calc dmu ntype: {ntype}")
LOGGER.info(f"Calc dmu parent: {parent}")

"""
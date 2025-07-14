import numpy as np

from utils._np.serialize_complex import deserialize_complex, check_serilisation


class FieldUtils:

    def __init__(self):
        self.i = 1j
        # 8 Gell-Mann-Matrizen (SU(3) Generatoren)
        self.lambda_1 = np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 0]], dtype=complex)

        self.lambda_2 = np.array([[0, -1j, 0],
                                  [1j, 0, 0],
                                  [0, 0, 0]], dtype=complex)

        self.lambda_3 = np.array([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 0]], dtype=complex)

        self.lambda_4 = np.array([[0, 0, 1],
                                  [0, 0, 0],
                                  [1, 0, 0]], dtype=complex)

        self.lambda_5 = np.array([[0, 0, -1j],
                                  [0, 0, 0],
                                  [1j, 0, 0]], dtype=complex)

        self.lambda_6 = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]], dtype=complex)

        self.lambda_7 = np.array([[0, 0, 0],
                                  [0, 0, -1j],
                                  [0, 1j, 0]], dtype=complex)

        self.lambda_8 = (1 / np.sqrt(3)) * np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, -2]], dtype=complex)
        self.su3_group_generators = [self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4, self.lambda_5,
                                     self.lambda_6, self.lambda_7, self.lambda_8]

        # SU(2) Generatoren (Pauli-Matrizen / 2)
        self.T1 = 0.5 * np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)

        self.T2 = 0.5 * np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=complex)

        self.T3 = 0.5 * np.array([
            [1, 0],
            [0, -1]
        ], dtype=complex)

        self.su2_group_generators = [
            self.T1,
            self.T2,
            self.T3
        ]
        ### GAMMA ###

        self.gamma_0 = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, -1]], dtype=complex)

        self.gamma_1 = np.array([[0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [-1, 0, 0, 0]], dtype=complex)

        self.gamma_2 = np.array([
            [0, 0, 0, -1j],
                            [0, 0, 1j, 0],
                            [0, 1j, 0, 0],
                            [-1j, 0, 0, 0]], dtype=complex)

        self.gamma_3 = np.array([[0, 0, 1, 0],
                            [0, 0, 0, -1],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=complex)

        self.gamma = [self.gamma_0, self.gamma_1, self.gamma_2, self.gamma_3]
        self.gamma5 = (
                self.i * self.gamma[0] @ self.gamma[1] @ self.gamma[2] @ self.gamma[3]
        )
        self.gamma0_inv = np.linalg.inv(self.gamma[0])  # (4,4) matrix


        self.fermion_to_gauge_couplings = {
            "electron": ["photon", "w_plus", "w_minus", "z_boson"],
            "myon": ["photon", "w_plus", "w_minus", "z_boson"],
            "tau": ["photon", "w_plus", "w_minus", "z_boson"],
            "electron_neutrino": ["w_plus", "w_minus", "z_boson"],
            "myon_neutrino": ["w_plus", "w_minus", "z_boson"],
            "tau_neutrino": ["w_plus", "w_minus", "z_boson"],
            "up_quark": ["photon", "w_plus", "w_minus", "z_boson",
                         "gluon"],
            "down_quark": ["photon", "w_plus", "w_minus", "z_boson", "gluon"],
            "charm_quark": ["photon", "w_plus", "w_minus", "z_boson", "gluon"],
            "strange_quark": ["photon", "w_plus", "w_minus", "z_boson", "gluon"],
            "top_quark": ["photon", "w_plus", "w_minus", "z_boson", "gluon"],
            "bottom_quark": ["photon", "w_plus", "w_minus", "z_boson", "gluon"],
        }

        self.fermion_fields = [
            # Leptonen
            "electron",  # ψₑ
            "myon",  # ψ_μ
            "tau",  # ψ_τ
            "electron_neutrino",  # νₑ
            "myon_neutrino",  # ν_μ
            "tau_neutrino",  # ν_τ

            # Quarks
            "up_quark",  # ψᵤ
            "down_quark",  # ψ_d
            "charm_quark",  # ψ_c
            "strange_quark",  # ψ_s
            "top_quark",  # ψ_t
            "bottom_quark"  # ψ_b
        ]

        self.g_qcd=[
            # Starke Wechselwirkung (SU(3)_C)
            "gluon",  # G 8x4 array
        ]

        self.g_qed = [
            # Elektromagnetismus (U(1)_Y → nach Mischung: Photon)
            "photon",  # A_μ
        ]

        self.g_electroweak = [
            # Schwache Wechselwirkung (SU(2)_L)
            "w_plus",  # W⁺
            "w_minus",  # W⁻
            "z_boson",  # Z⁰
        ]

        self.gauge_fields = [
            *self.g_qcd,
            *self.g_electroweak,
            *self.g_qed
        ]

        self.gauge_to_gauge_couplings = {
            "gluon": ["gluon"],

            # Elektroschwach
            "w_plus": ["w_minus", "z_boson", "photon"],
            "w_minus": ["w_plus", "z_boson", "photon"],
            "z_boson": ["w_plus", "w_minus", "photon"],
            "photon": ["w_plus", "w_minus", "z_boson", "photon"],

            # GUT/erweiterte Modelle (optional, falls du auch sowas simulierst)
            #"b_boson": ["w_plus", "w_minus", "z_boson", "gluon"],  # z. B. in GUT wie SU(5)

            # Gravitationsähnlich (selten in QFT-Sim enthalten)
            "graviton": ["photon", "gluon", "w_plus", "w_minus", "z_boson", "graviton"]
        }

        self.g_h_couplings = [
            "w_plus",
            "w_minus",
            "z_boson",
        ]
        self.gauge_to_fermion_couplings = {
            "photon": [
                "electron", "myon", "tau",
                "up_quark", "down_quark", "charm_quark",
                "strange_quark", "top_quark", "bottom_quark"
            ],
            "w_plus": [
                "electron", "myon", "tau",
                "electron_neutrino", "myon_neutrino", "tau_neutrino",
                "up_quark", "down_quark", "charm_quark",
                "strange_quark", "top_quark", "bottom_quark"
            ],
            "w_minus": [
                "electron", "myon", "tau",
                "electron_neutrino", "myon_neutrino", "tau_neutrino",
                "up_quark", "down_quark", "charm_quark",
                "strange_quark", "top_quark", "bottom_quark"
            ],
            "z_boson": [
                "electron", "myon", "tau",
                "electron_neutrino", "myon_neutrino", "tau_neutrino",
                "up_quark", "down_quark", "charm_quark",
                "strange_quark", "top_quark", "bottom_quark"
            ],
            "gluon": [
                "up_quark", "down_quark", "charm_quark",
                "strange_quark", "top_quark", "bottom_quark"
            ]
        }

        self.all_sub_fields = [
            *[k.upper() for k in self.gauge_fields],
            *[k.upper() for k in self.fermion_fields],
            "PHI"
        ]

        # 4 array
        self.leptons = [
            "electron",
            "myon",
            "tau",
            "electron_neutrino",  # νₑ
            "myon_neutrino",  # ν_μ
            "tau_neutrino",  # ν_τ
        ]

        # Have color charge and interat with gluons -> psi is different fro leptons
        self.quarks = [
            "up_quark",
            "down_quark",
            "charm_quark",
            "strange_quark",
            "top_quark",
            "bottom_quark"
        ]
        self.gauge_groups = {
            "U(1)_Y": ["b_boson"],  # Hypercharge
            "SU(2)_L": ["w_boson_1", "w_boson_2", "w_boson_3"],  # Schwache Isospin
            "SU(3)_C": ["gluon"],  # Farbe
            "U(1)_EM": ["photon"],  # Nach SSB (Mischung aus B und W₃)
            "Electroweak (SU(2)_L × U(1)_Y)": ["w_boson_±", "z_boson", "photon"]  # nach SSB
        }
        self.symmetry_groups = {
            "U(1)_Y": [
                "photon",  # nach SSB Teil der Mischung
                "Z_boson",  # ebenfalls Teil der Mischung
                "electron",
                "myon",
                "tau",
                "electron_neutrino",  # νₑ
                "myon_neutrino",  # ν_μ
                "tau_neutrino",  # ν_τ
                *self.quarks,
                "phi"
            ],
            "SU(2)_L": [
                "W_plus",
                "W_minus",
                "W_boson_3",
                "electron",
                "muon",
                "tau",
                "neutrino_e",
                "neutrino_mu",
                "neutrino_tau",
                "up_quark",
                "down_quark",
                "charm_quark",
                "strange_quark",
                "top_quark",
                "bottom_quark"
                "phi"
            ],
            "SU(3)_C": [
                *self.quarks,
                "gluon"
            ]
        }
        
    def _field_value(self, type):
        type = type.lower()
       #print("_field_value", type)
        key=None
        if type in ["gluon", "gluon_item"]:
            key= "G"
        if type == "z_boson":
            key= "Z"
        #if type == "b_boson":
        #    return "B"
        if type == "w_plus":
            key= "Wp"
        if type == "w_minus":
            key= "Wm"
        if type == "photon":
            key= "A"
       #print("key=", key)
        return key

    def _convert_to_complex(self, com):
        """
        Wandelt Listen oder serialisierte Dicts in ein komplexes NumPy-Array um.
        """
        # Wenn es ein Dict ist, deserialisieren
        if isinstance(com, dict) and "bytes" in com:
            com = deserialize_complex(com)

        # Wenn es schon ein NumPy-Array ist, direkt zurückgeben
        if isinstance(com, np.ndarray):
            return com

        # Sonst weiterverarbeiten wie bisher
        complex_rows = []
        for row_data in com:
            complex_row = [complex(re, im) for re, im in row_data]
            complex_rows.append(complex_row)

        psi = np.array(complex_rows, dtype=complex)
        return psi

    def get_sym_group(self, ntype):
        all_active_groups = []
        for group, ntypes in self.symmetry_groups.items():
            if ntype.lower() in [n.lower() for n in ntypes]:
                all_active_groups.append(group)

       #print(f"Sym Groups for {ntype} set: {all_active_groups}")
        if not len(all_active_groups):
            raise ValueError(f"No SymGroup for {ntype}")
       #print("get_sym_group", all_active_groups)

        return all_active_groups



    def check_serialize_dict(self, data, attr_keys):
        new_dict={}
        for k, v in data.items():
            if k in attr_keys:
               #print("Convert sd key:", k, type(v), v)
                v = check_serilisation(v)
                new_dict[k] = v
        return new_dict


    def restore_selfdict(self, data):
        new_dict = {}
        #print("restore_selfdict data B4:", data)
        if isinstance(data, dict):
            #print(f"Restore data for {data.get('id')}")

            for k, v in data.items():
                if isinstance(v, dict) and "serialized_complex" in v:  # ("bytes" in v or "data" in v)
                    v = deserialize_complex(
                        bytes_struct=v,
                    )
                #v = convert_numeric(v)
                new_dict[k] = v
        """elif isinstance(data, list):
            new_dict = [self.restore_selfdict(item) for item in data]
        """
        #print("new_dict:", new_dict)
        return new_dict

    def get_g_V(self, isospin, charge, sin2W=0.231):
        """print("get_g_V")
        #print("charge", charge)
        #print("isospin", isospin)"""
        g_V = float(isospin) - 2 * float(charge) * sin2W
        return g_V






    def get_o_operator(self, ntype, g_V=None, isospin=None, charge=None, gluon_index=None):
        #print("get_o_operator")
        ntype = ntype.lower()
        if ntype in ["z_boson", "w_plus", "w_minus"]:
            if g_V is None or isospin is None:
                raise ValueError("Für Z müssen g_V, isospin und gamma5 gesetzt sein.")
            return g_V * np.eye(4, dtype=complex) - isospin * self.gamma5

        elif ntype == "photon":
            return charge * np.eye(4, dtype=complex)

        elif ntype == "gluon":
            if gluon_index is None:
                raise ValueError("Für Gluon muss su3_generator gesetzt sein.")
            return self.su3_group_generators[gluon_index]

        else:
            raise ValueError(f"Unbekannter Feld-Typ '{ntype}'")











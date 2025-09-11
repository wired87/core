
import numpy as np
import ray

from qf_core_base.fermion.ferm_utils import FermUtils
from qf_core_base.ray_validator import RayValidator

from utils._np.serialize_complex import check_serialize_dict, deserialize_complex
from utils.graph.local_graph_utils import GUtils

class FermionBase(
    FermUtils,
    RayValidator
):
    """
    todo: Bewegungsgleichungen ?

    🔹 1️⃣ Fermionfelder
    Das sind Materiefelder.

    Sie bilden ALLE bekannten Teilchen (Elektronen, Quarks, Neutrinos).

    Ohne Fermionfelder: keine Materie, kein Aufbau.
    ✅ Fermionfelder sind fundamental als Träger der Materie.

    links und rechtshändige fermionen
    """

    def __init__(
            self,
            d,
            attr_keys,
            theta_W,
            g:GUtils=None,
            host:dict=None,
    ):
        FermUtils.__init__(self)
        RayValidator.__init__(self, host, g)
        self.dirac_result = None
        self.handedness = None
        self.short_lower_type = None
        self.is_quark = None
        self.neighbor_pm_val_same_type = None
        self.attrs = None
        self.type = None
        self.all_subs = None
        self.env = None
        self.isospin = None
        self.dpsi = None

        # args
        self.attr_keys = attr_keys
        self.theta_W = theta_W
        self.d = d

        # locals
        self.quark_type_coupling_check = ["up", "charm", "top"]

    def main(
            self,
            env,
            attrs,
            all_subs,
            neighbor_pm_val_same_type,
            **kwargs
    ):
        # args
        self.env = env
        self.all_subs=all_subs
        self.attrs = self.restore_selfdict(attrs)
        self.type = self.attrs.get("type")
        self.neighbor_pm_val_same_type = neighbor_pm_val_same_type

        # attrs
        for k, v in self.attrs.items():
            setattr(self, k, v)

        #quark
        if self.type:
            self.is_quark = self._is_quark(self.type)

        # extract lower pre type for quarks (up, down, etc.)
        self.short_lower_type = self.type.lower().split("_")[0]

        # prev
        if self.attrs.get("psi_prev") is None:
            self.attrs["psi_prev"] = self.psi.copy()
        new_prev_psi = self.psi.copy()

        # spin
        self.handedness = self.check_spin(getattr(self, "psi", None), self.is_quark)
        self.isospin = self._isospin(
            self.handedness,
            ntype=self.type
        )
        
        self.attrs["psi_bar"] = self._psi_bar(self.psi.copy(), self.is_quark)
        self._coupling_term()
        self._dpsi()
        self._dirac_process()
        self._psi()

        # convert back
        self.dpsi = None

        self.attrs["psi_prev"] = new_prev_psi

        new_dict = check_serialize_dict(
            self.__dict__,
            self.attr_keys
        )

        return new_dict

    def _dpsi(self):
        # 📌 Die Dirac-Gleichung kombiniert alle berechneten Kopplungsterme
        self.dpsi = self.call(
            method_name="_dX",
            attrs=self.attrs,
            d=self.d,
            neighbor_pm_val_same_type=self.neighbor_pm_val_same_type,
            field_key="psi"
        )
        print(f"∂μψ(x): {self.dpsi}")


    def _dirac_process(self):
        # preset dirac resutl
        if getattr(self, "dirac_result", None) is None:
            self.dirac_result = np.zeros(self.psi.shape, dtype=complex)

        spatial = np.zeros_like(self.psi, dtype=complex)
        d_psi_copy = self.dpsi.copy()
        for i, dmu_psi in enumerate(d_psi_copy):
            if self.is_quark is True:
                print("Quark")
                spatial += self._dirac(
                    i, dmu_psi.copy()
                )
            else:
                print("!Quark")
                spatial += self._dirac(i, dmu_psi.copy())

        #print("self.psi", self.psi, type(self.psi), self.psi.shape)
        #print("self.mass", self.mass, type(self.mass))
        #print("self.i", self.i, type(self.i))

        if self.is_quark is True:
            for i in range(3):
                # Massenterm abziehen
                mass_term = -self.i * self.mass * self.psi[i]
                print(f"mass_term: {mass_term}")
                self.dirac_result[i] = -self.gamma0_inv @ (spatial[i] + mass_term)

        else:
            # Massenterm abziehen
            mass_term = -self.i * self.mass * self.psi
            print(f"imψ = {mass_term}")
            self.dirac_result = -self.gamma0_inv @ (spatial + mass_term)
        print(f"(iγμ∂μ−m)ψ(x) = {self.dirac_result}")

    def _dirac(self, i, dmu_psi):
        """
        Time evolution (for sim)
        Uses Kinetic derivation & coupling term of neighbor gauges
        """
        # t,x,y,z
        gamma_item = self.gamma[i]
        for ntype, neighbor_struct in self.all_subs["GAUGE"].items():
            for gid, nattrs in neighbor_struct.items():
                # todo improve
                edge_attrs:dict= None
                for eid, eattrs in self.all_subs["edges"].items():
                    if self.id in eid and gid in eid:
                        edge_attrs = eattrs
                        break

                if edge_attrs is not None:
                    # coupling_term # 4x txyz
                    coupling_term = edge_attrs.get("coupling_term")
                    if coupling_term is not None:
                        coupling_term = self.extract_coupling_from_quark_doublet(
                            coupling_term
                        )

                        # sum copling termitem for given direction (t,x,y or z)
                        #print(f"coupling_term: {coupling_term}-{coupling_term.shape}")
                        dmu_psi = self.sum_coupling_dpsi(
                            coupling_term,
                            dmu_psi
                        )
                    else:
                        #print(f"NEED TO CALC COUPLING TERM BETWEEN {self.id} AND {gid}")
                        pass
        return self._calc_dirac_result(
            gamma_item,
            dmu_psi
        )

    def calc_time(self):
        all_edges = ray.get(self.host["UTILS_WORKER"].call.remote(
            method_name="get_edges_from_node",
            nid=self.id
        ))

        sum_coupling_terms = 0
        for edge in all_edges:
            if "coupling_term" in edge:
                sum_coupling_terms += edge["coupling_term"]
        _time = self.psi + sum_coupling_terms



    def dmu_time(self):
        pass

    def _calc_dirac_result(self, gamma_item, dmu_psi):
        if self.is_quark:
            result = np.zeros((3, 4), dtype=complex)
            for i in range(3):
                result[i] = gamma_item @ dmu_psi[i]
        else:
            result = gamma_item @ dmu_psi
        #print(f"dmu_psi: {result, result.shape}")
        return result

    def extract_coupling_from_quark_doublet(self, coupling_term):
        """
        Extract a singlet from doublet quark -> w+- coupling from given type
        """
        if coupling_term.shape == (3, 2, 4):
            #print(f"Doublet coupling term: {coupling_term}")
            part_coupling_term = np.zeros((3, 4), dtype=complex)
            if self.short_lower_type in self.quark_type_coupling_check:
                item = 0
            else:
                item = 1

            for i in range(3):
                part_coupling_term[i] = coupling_term[i][item]

            #print("Singlet coupling term extracted")
            coupling_term = part_coupling_term

        return coupling_term


    #########################



    def _psi(self):
        """
        ψ(t+Δt) = ψ + Δt · ∂tψ
        """
        m = float(getattr(self, "mass", None))
        print(f"m: {m}")
        # print(f"dpsi_dt", dpsi_dt)

        # ENDLICH: ψ = ψ + Δt · ∂tψ
        #dpsi_dt = self.gamma0_inv @ self.dirac_result  # ∂tψ = γ⁰⁻¹ · Dirac-Term
        self.psi = self.psi + self.d["t"] * self.dirac_result  # ψ = ψ + Δt · ∂tψ
        print(f"ψ+Δt·∂tψ: {self.psi}")


    def _coupling_term(self):
        """
        Loop through g neighbors of self and calcs coupling strength for all
        fermions interagieren niemals untereinander
        """
        nid = getattr(self, "id", None)
        psi = getattr(self, "psi").copy()
        #print("self.type", self.type)
        try:
            for ntype, neighbor_struct in self.all_subs["GAUGE"].items():
                for nnid, nattrs in neighbor_struct.items():
                    # Gauge attrs
                    field_value= deserialize_complex(nattrs.get(self._field_value(ntype)))
                    g = nattrs.get("g")
                    nntype = nattrs.get("type")
                    gluon_index = nattrs.get("gluon_index", None)  # for gluon calc

                    # coupling schema (diff for quark w+/-)
                    gauge_coupling_schema = self.gauge_coupling_schema(nntype)

                    # Form doublets for quark -> w+/-
                    if nntype.lower() in ["w_minus", "w_plus"]:
                        self.w_coupling(
                            field_value,
                            g,
                            ntype,
                            gluon_index,
                            gauge_coupling_schema,
                            psi,
                        )

                    elif nntype.lower() == "z_boson":
                        gauge_coupling_schema = self.z_coupling_process(
                            psi,
                            field_value,
                            g,
                            ntype,
                            gluon_index,
                            gauge_coupling_schema
                        )

                    else:
                        gauge_coupling_schema = self._calc_coupling_term_G(
                            psi, field_value, g, ntype, gluon_index, gauge_coupling_schema, is_quark=self.is_quark,
                            handedness=self.handedness
                        )
                    print(f"J(x): {gauge_coupling_schema}")
                    self.add_coupling_term(
                        nid,
                        nnid,
                        gauge_coupling_schema,
                    )

            for ntype, neighbor_struct in self.all_subs["PHI"].items():
                for nnid, nattrs in neighbor_struct.items():
                    # Coupling
                    coupling_shema_yukawa = self.get_yukawa_coupling_schema()
                    yukawa_total_coupling = self._yukawa_couping_process(
                        nattrs,
                        coupling_shema_yukawa,
                        self.is_quark,
                        getattr(self, "y"),
                        self.attrs
                    )

                    # Update edge coupling term
                    self.add_coupling_term(
                        nid,
                        nnid,
                        yukawa_total_coupling,
                    )
        except Exception as e:
            print(f"Error calc _coupling_term: {e}")

    def w_coupling(
            self,
            field_value,
            g,
            ntype,
            gluon_index,
            gauge_coupling_schema,
            psi,

    ):
        if self.is_quark is True and self.short_lower_type in self.quark_type_coupling_check:
            """
            Coupling happens just from u-type quarks(up,top,charm) 
            -> form dublet 
            -> etract left side (w+- jsut couples on left)
            -> and bring the structue back again
            """
            psi = self.get_quark_doublet()
            #print("Extracted quark dublet")
            gauge_coupling_schema = np.zeros_like(psi, dtype=complex)
            for i in range(3):  # Loop über Farbindizes
                for j in range(2):  # Loop über das Dublett (Up oder Down)
                    # Hier ist der Dirac-Spinor (3x4), den Sie an die Methode übergeben müssen
                    single_dirac_spinor = psi[i][j]
                    left_side_psi = self.extract_psi_lrm(single_dirac_spinor, handedness="left", is_quark=False) # working here single spinor
                    gauge_coupling_schema[i][j][:2] += self.fermion_gauge_coupling(
                        left_side_psi,  # without time
                        field_value,
                        g,
                        ntype,
                        gluon_index,
                        handedness="left",
                    )


        elif self.is_quark is False:
            # Extract only left handed part
            left_side_psi = self.extract_psi_lrm(psi, handedness="left", is_quark=False)  # working here single spinor
            #print(f"Extracted left handed spinor: {left_side_psi}")
            gauge_coupling_schema[:2] += self.fermion_gauge_coupling(
                left_side_psi,  # without time
                field_value,
                g,
                ntype,
                gluon_index,
                handedness="left",
            )
        return gauge_coupling_schema

    def gauge_coupling_schema(self, gauge_type=None):
        try:

            #print("gauge_coupling_schema gauge_type", gauge_type, self.psi.shape, self.is_quark)
            if self.is_quark and gauge_type.lower() in ["w_plus", "w_minus"]:
                #print(f"Quark -> {gauge_type} detected")
                return np.zeros((3, 2, 4), dtype=complex)
            else:
                #print("Get efault coupling schema ")
                return np.zeros_like(self.psi, dtype=complex)
        except Exception as e:
            print(f"Error get gauge coupling sh´hema: {e}")


    def z_coupling_process(
            self,
            psi,
            field_value,
            g,
            ntype,
            gluon_index,
            gauge_coupling_schema
    ):
        # z couples to l/r AND quark
        #split_psi = self.extract_psi_lrm(psi, self.handedness, self.is_quark)
        # z boson ist farbenblind. es genügt einen spinor zu berechenn und diesen mit 3 (für alle farben) zu summieren

        if self.is_quark is True:
            for i in range(3):
                gauge_coupling_schema[i] += self.calc_coupling_lr(
                    psi[i], field_value, g, ntype, gluon_index, gauge_coupling_schema[i]
                )
        else:
            gauge_coupling_schema = self.calc_coupling_lr(
                psi, field_value, g, ntype, gluon_index, gauge_coupling_schema
            )
        return gauge_coupling_schema



    def calc_coupling_lr(self, psi, field_value, g, ntype, gluon_index, gauge_coupling_schema):
        """
        Extract l/r part of the dirac spinor, and calculate the coupling for both sides. the result gets summed
        todo just adapt z boson genrator to handle full spinors per run
        """
        right_psi = self.extract_psi_lrm(psi, handedness="right", is_quark=False)
        left_psi = self.extract_psi_lrm(psi, handedness="left", is_quark=False)
        gauge_coupling_schema[2:] += self.fermion_gauge_coupling(
            right_psi,
            field_value,
            g,
            ntype,
            gluon_index,
            handedness="right",
        )
        #print(f"Calculated right coupling: {gauge_coupling_schema, gauge_coupling_schema.shape}")
        gauge_coupling_schema[:2] += self.fermion_gauge_coupling(
            left_psi,
            field_value,
            g,
            ntype,
            gluon_index,
            handedness="left",
        )
        #print(f"TOTAL Z COUPLING: {gauge_coupling_schema}")
        return gauge_coupling_schema



    def sum_coupling_dpsi(self, coupling, dpsi):
        """
        If ferm just coupled to one side of gauge,
        dpsi need toincrease on exactly this side...
        """
        #print(f"DPSI before: {dpsi}")
        shape = coupling.shape
        #print(f"coupling_shape: {coupling.shape}")
        if shape == (2, 1) and self.handedness == "left":
            dpsi[:2] += coupling
        if shape == (2, 1) and self.handedness == "right":
            dpsi[2:] += coupling
        else:
            dpsi += coupling
        #print(f"DPSI after: {dpsi}-{dpsi.shape}")
        return dpsi

    def add_coupling_term(
            self,
            nid,
            nnid,
            gauge_total_coupling,
    ):
        # Update edge coupling term
        if gauge_total_coupling is not None:
            for k, v in self.all_subs["edges"].items():
                #pprint.pp(v)
                if nid in k and nnid in k:
                    v.update(
                        {"coupling_term": gauge_total_coupling}
                    )



    def _yukawa_couping_process(self, nattrs, yukawa_total_coupling, is_quark, y, attrs):
        #print("nattrs", nattrs)
        if is_quark:
            for i in range(3):
                yukawa_term = self.yukawa_term(
                    y,
                    nattrs["h"],
                    getattr(self, "psi_bar"),
                    getattr(self, "psi"),
                    is_quark
                )
                yukawa_total_coupling[i] += yukawa_term[i]

        else:
            yukawa_term = self.yukawa_term(
                y,
                nattrs["h"],
                getattr(self, "psi_bar"),
                getattr(self, "psi"),
                is_quark
            )
            yukawa_total_coupling = yukawa_term
        #print(f"# _yukawa_couping_process finished: {yukawa_total_coupling}")
        return yukawa_total_coupling


    def get_yukawa_coupling_schema(self):
        return np.zeros_like(self.psi, dtype=complex)


    def gauss(self, x, mu=0, sigma=5):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


    def get_quark_doublet(self):
        """
        Erstellt ein valides (3, 2, 4) Quark-Dublett ψ
        für W±-Kopplung basierend auf dem aktuellen Quarktyp (ntype)
        todo untersceide zwischen l/r -> w kuppelt nur an l
        """
        partner_map = {
            "up": "down",
            "charm": "strange",
            "top": "bottom",
        }

        if self.short_lower_type not in partner_map:
            raise ValueError(f"Kein gültiger Quarktyp für W-Kopplung: {self.short_lower_type}")

        # eigener zustand
        psi_self = getattr(self, "psi")  # erwartet shape (3, 4)
        #print("psi_self", psi_self)

        # Quark kupelt nur an linke seite (if links: rechts = 0 -> nimm kompletten spinor)
        # partner-nid ermitteln
        self_nid= getattr(self, "id")
        #print("self_nid", self_nid)

        # alltimes 3,4
        total_down_psi_sum = np.zeros_like(self.psi, dtype=complex)

        ckm_struct = self.ckm[self.short_lower_type]  # e.g. {"d": 0.974, "s": 0.225, "b": 0.004}
        for quark_type, ckm_val in ckm_struct.items():
            # Extract Neighbor

            quark_type = f"{quark_type}_quark".upper()
            #print(f"get_quark_doublet from {quark_type}")
            neighbor_quark = self.all_subs["FERMION"][quark_type]  # dict: id:attrs

            # Extract Data tuple: id, attrs & args
            item_paare = list(neighbor_quark.items())[0]
            item_attrs = item_paare[1]

            neighbor_psi = deserialize_complex(
                    item_attrs.get("psi")
                )

            # Get & Check handedness
            n_handedness = item_attrs.get("handedness", None)

            if n_handedness and isinstance(n_handedness, str) and n_handedness == "left":
                # Sum CKM val and neighbor_quark_psi
                component = neighbor_psi * ckm_val

            else:
                # Default value für right handed Quarks
                component = 0

            # Add to total
            total_down_psi_sum += component

        #print(f"psi_self: {psi_self}-{psi_self.shape}")
        #print(f"total_down_psi_sum: {total_down_psi_sum}")
        doublet = np.stack([psi_self, total_down_psi_sum], axis=1)
        #print(f"Quark doublet extracted: {doublet, doublet.shape}")

        return doublet



"""

        if isinstance(split_psi, tuple):
            print("Start wroking mixed handedness psis")
            # self. handedness == mixed ->
            # need to calc L/R sepparate and sum coupling term
            coupling_term = np.zeros_like(self.psi, dtype=complex)
            
            for i, side_psi in enumerate(split_psi):
                print(f"calc coupling with {side_psi, side_psi.shape}")
                coupling = self._calc_coupling_term_G(
                    side_psi, field_value, g, ntype, gluon_index, gauge_coupling_schema, is_quark=self.is_quark
                )

                # Apply both single couplings to term spinor
                if coupling_term is not None:
                    if i == 0:
                        # left side
                        coupling_term[:2] += coupling
                    else:
                        # r
                        coupling_term[2:] += coupling

"""
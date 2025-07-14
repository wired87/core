
import numpy as np

from qf_core_base.fermion.ferm_utils import FermUtils
from qf_core_base.qf_utils.all_subs import G_FIELDS
from qf_core_base.qf_utils.qf_utils import QFUtils
from qf_core_base.symmetry_goups.main import SymMain

from utils._np.serialize_complex import check_serialize_dict, deserialize_complex
from utils.graph.local_graph_utils import GUtils


class FermionBase(FermUtils):
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
            g: GUtils,
            qfn_id,
            d,
            neighbors_pm,
            attr_keys,
            attrs,
            theta_W,
            time=0.0,
            **args
    ):
        super().__init__()
        # Deserialize attrs
        self.attrs = self.restore_selfdict(attrs)
        # make class attrs
        # LOGGER.info("init FermionBase")
        for k, v in self.attrs.items():
            setattr(self, k, v)
            # LOGGER.info(f"{k}:{v}")


        # Validate sib_type (case quark -> split into 3 pcs)
        sub_type = getattr(self, "sub_type")
        #if sub_type == "ITEM":
        #if sub_type == "BUCKET":
            # Todo outsrc tasks to bucekt (like total quark lagrangian,...)
        # return



        self.g = g
        if self.type:
            self.is_quark = self._is_quark(self.type)
        # LOGGER.info("Is Quark", self.is_quark, self.type, self.psi.shape)
        self.attr_keys = attr_keys
        #self.time = time
        self.dpsi = None
        self.qf_utils = QFUtils(g)
        self.qfn_id = qfn_id
        self.theta_W = theta_W
        self.d = d
        self.parent = self.parent[0].lower()
        self.neighbors_pm = neighbors_pm

        # Isospin
        ntype = getattr(self, "type", None)
        handedness = self.check_spin(getattr(self, "psi", None), self.is_quark)
        self.isospin = self._isospin(
            handedness,
            ntype
        )

        # LOGGER.info("self.symmetry_groups", self.attrs["_symmetry_groups"])

        self.symmetry_group_class = SymMain(groups=self.attrs["_symmetry_groups"][0])
        self.neighbors = self.g.get_neighbor_list(
            self.attrs["id"],
            self.all_sub_fields
        )

        self.same_type_neighbors = []

        """for n in self.neighbors:
            # LOGGER.info(f"{self.id} neighbor:", n[0], n[1])
        """
        # Get same type neighbors
        for k, (p, m) in self.neighbors_pm.items():
            self.same_type_neighbors.append(
                self.g.get_single_neighbor_nx(p, self.attrs["type"].upper())
            )
            self.same_type_neighbors.append(
                self.g.get_single_neighbor_nx(m, self.attrs["type"].upper())
            )

    def main(self):
        # LOGGER.info("Fermion main")
        # LOGGER.info("Raw fermion:", self.attrs)
        if self.attrs["psi_prev"] is None:
            self.psi_prev = self.psi.copy()
        new_prev_psi = self.psi.copy()

        self.attrs["psi_bar"] = self._psi_bar(self.psi.copy(), self.is_quark)

        self._dpsi(new_prev_psi)
        self._dmu_psi()
        self._kinetic_term()
        self._coupling_term()

        # convert back
        self.dpsi = None

        self.psi_prev = new_prev_psi

        self.parent = [self.parent.upper()]

        new_dict = check_serialize_dict(
            self.__dict__,
            self.attr_keys
        )

        # LOGGER.info("Updated fermion:", new_dict)
        self.g.update_node(new_dict)
       #print(f"Update for {self.id} finished")
        # LOGGER.info(f"finiehsed update of {self.id}: {new_dict}")


    def _dmu_psi(self):
        # todo impr differ between quark and leptons
        dmu_psi = []
        for mu in range(4):  # t, x, y, z
            """if self.is_quark:
                # self.gamma[mu] is (4,4), self.dpsi[rgb] is (3,4)
                dpsi_item = self.dpsi[mu]
                # LOGGER.info("Quark dpsi_item", dpsi_item)

                result = np.einsum("jk,ak->aj", self.gamma[mu], dpsi_item)
                # LOGGER.info(f"Quark dmu_psi item result: {result}{result.shape}")
                dmu_psi.append(result)
                ## LOGGER.info("self.dmu_psi", self.dmu_psi)
            else:"""
            # self.gamma[mu] is (4,4), self.dpsi[mu] is (4,1)
            dpsi_item = self.dpsi[mu]
            # LOGGER.info("Quark dpsi_item", dpsi_item, dpsi_item.shape)
            result = np.dot(self.gamma[mu], dpsi_item)
            # LOGGER.info(f"Quark dmu_psi item result: {result}{result.shape}")

            dmu_psi.append(result)
            ## LOGGER.info("self.dmu_psi", self.dmu_psi)

        self.dmu_psi = dmu_psi
        # LOGGER.info("ψ:", dmu_psi)





    def _single_dpsi(self, psi_forward, psi_backward, d, time=False):
        """
        Calcs single dpsi - entries -> splits spinors for quarks
        """
        if time is False:
            # convert neighbor psi to complex
            # # LOGGER.info("psi_forward, psi_backward", psi_forward, psi_backward)
            psi_forward = np.array(psi_forward, dtype=complex)
            psi_backward = np.array(psi_backward, dtype=complex)

            #psi_forward=self._convert_to_complex(psi_forward)
            #psi_backward=self._convert_to_complex(psi_backward)
            # LOGGER.info("psi_forward", psi_forward)
            # LOGGER.info("psi_backward", psi_backward)
        else:
            # psi already complex
            pass
        # # LOGGER.info("psi_forward", type(psi_forward), psi_forward)
        # # LOGGER.info("psi_backward", type(psi_backward), psi_backward)
        # # LOGGER.info("dX", (psi_forward - psi_backward) / (2 * d))
        """if self.is_quark:
            single_dpsi = []
            for i in range(3):
                dpsi_item = (psi_forward[i] - psi_backward[i]) / (2 * d)
                single_dpsi.append(dpsi_item)
        else:"""
        single_dpsi = (psi_forward - psi_backward) / (2 * d)
        return single_dpsi

    def _dpsi(self, new_prev_psi):
        """
        # LOGGER.info("neighbors_pm", self.neighbors_pm)
        # LOGGER.info("self.psi", self.psi, self.psi.shape)
        # LOGGER.info("self.psi_prev", new_prev_psi)
        # LOGGER.info("self.psi_prev.shae", new_prev_psi.shape)
        """
        psi_t = self._single_dpsi(self.psi, new_prev_psi, self.d["t"], time=True)  # dt = timestep
        dpsi = [
            psi_t
        ]
        ## LOGGER.info("neighbors", self.neighbors_pm)
        for i, (key, pm) in enumerate(self.neighbors_pm.items()):  # x,y,z
            # LOGGER.info("DPSI run", i)
            plus_id = pm[0]
            minus_id = pm[1]

            neighhbor_plus = self.g.get_single_neighbor_nx(plus_id, self.type.upper())[1]
            neighhbor_minus = self.g.get_single_neighbor_nx(minus_id, self.type.upper())[1]

            psi_plus = deserialize_complex(neighhbor_plus[self.parent])
            # LOGGER.info(f"nplus {minus_id}-> {neighhbor_plus[1]}:{psi_plus}")

            psi_minus = deserialize_complex(neighhbor_minus[self.parent])
            # LOGGER.info(f"nminus {plus_id}-> {neighhbor_minus[1]}:{psi_minus}")

            if psi_plus is None:
                psi_plus = self.psi
            if psi_minus is None:
                psi_minus = self.psi

            dpsi_x = self._single_dpsi(psi_plus, psi_minus, self.d[key])
            # LOGGER.info(f"dpsi{i}", dpsi_x)
            dpsi.append(dpsi_x)  # -> alle koords
            # LOGGER.info("Quark:", self.is_quark)
        # LOGGER.info("Finished dpsi", dpsi)
        self.dpsi = dpsi


    def _kinetic_term(self):
        """if self.is_quark:
            self.kinetic_energy = sum([
                item for item in self.dmu_psi
                #if item.shape == (3, 4)
            ])
        else:"""
        # Lepton: (4,1) Arrays → einfache Summierung
        # LOGGER.info("Dmu_psi raum: ", self.dmu_psi[1:])
        for dp in self.dmu_psi:
            # LOGGER.info("lepton dpsi raum:", dp, dp.shape)
            pass
        self.kinetic_energy = np.sum(self.dmu_psi, axis=0)

        # LOGGER.info("self.kinetic_energy set:", self.kinetic_energy)


    def _psi(self, interaction_term):
        """
        Berechnet ψ(t+Δt) = ψ + Δt · ∂tψ mit Hilfe der Dirac-Gleichung.
        """
        m = float(getattr(self, "mass", None))
        """if self.is_quark is True:
            for i in range(3):
                rhs_term = -self.i * (self.kinetic_energy - m * self.psi[i] - interaction_term[i])
                # LOGGER.info(f"rhs_term quark", rhs_term)
                dpsi_dt = self.gamma0_inv @ rhs_term[i]
                # LOGGER.info(f"dpsi_dt", dpsi_dt)
                self.psi[i] = self.psi[i] + self.d["t"] * dpsi_dt
        else:"""
        # LOGGER.info("self.i", self.i)
        # LOGGER.info("self.kinetic_energy", self.kinetic_energy)
        # LOGGER.info("psi", self.psi)
        # LOGGER.info("interaction_term", interaction_term)
        # LOGGER.info("m", m)

        rhs_term = -self.i * (self.kinetic_energy - m * self.psi - interaction_term)
        # LOGGER.info(f"rhs_term lepton", rhs_term)

        dpsi_dt = self.gamma0_inv @ rhs_term
        # LOGGER.info(f"dpsi_dt", dpsi_dt)

        # ENDLICH: ψ = ψ + Δt · ∂tψ
        self.psi = self.psi + self.d["t"] * dpsi_dt
        # LOGGER.info(f"ψ = {self.psi}")


    def _coupling_term(self):
        """
        Loop through g neighbors of self and calcs coupling strength for all
        fermions interagieren niemals untereinander
        """
        nid = getattr(self, "id", None)
        psi = getattr(self, "psi")


        # LOGGER.info("_coupling_term for", nid)
        yukawa_total_coupling, gauge_total_coupling = self._get_coupling_schema()
        for n in self.neighbors:
            nnid = n[0]
            nattrs = n[1].copy()
            ntype = nattrs.get("type")

            if ntype in G_FIELDS:
                # get symmetry group from neighbor
                field_value= nattrs.get(self._field_value(ntype))
                g = nattrs.get("g")

                gauge_total_coupling = self._calc_coupling_term_G(
                    psi,
                    field_value,
                    g,
                    ntype,
                    gauge_total_coupling,
                    is_quark=self.is_quark
                )

                # Update edge coupling term
                if gauge_total_coupling is not None:
                    self.g.update_edge(
                        src=nid,
                        trgt=nnid,
                        attrs=self.attrs.update(
                            {"coupling_term": gauge_total_coupling}
                        )
                    )

            elif ntype == "PHI":
                # LOGGER.info(f"process psi {self.type} & phi")
                yukawa_total_coupling=self.symmetry_group_class._yukawa_couping_process(
                    nattrs,
                    yukawa_total_coupling,
                    self.is_quark,
                    getattr(self, "y"),
                    self.attrs
                )

                # Update edge coupling term
                if yukawa_total_coupling is not None:
                    self.g.update_edge(
                        src=nid,
                        trgt=nnid,
                        rels=["intern_coupled", "extern_coupled"],
                        attrs={"coupling_term": yukawa_total_coupling}
                    )

        interaction = self._merge_coupling_terms(
            yukawa_total_coupling,
            gauge_total_coupling
        )

        # LOGGER.info("_coupling term finished")
        self._psi(interaction)




    def _merge_coupling_terms(self, gauge_acc, yukawa_acc):

        if self.is_quark:
            total_interaction = []
            for i in range(3):
                merge_result = gauge_acc[i] + (yukawa_acc[i] * self.psi[i])
                total_interaction.append(merge_result)
        else:
            total_interaction = gauge_acc + (yukawa_acc * self.psi)

        # LOGGER.info("# _merge_coupling_terms finished:", total_interaction)
        return total_interaction



    def _get_coupling_schema(self):
        if self.is_quark:
            yukawa_acc = [0.0j * np.ones_like(self.psi[0]) for _ in range(3)]
            gauge_acc = [0.0j * np.ones_like(self.psi[0]) for _ in range(3)]
        else:
            yukawa_acc = 0.0j * np.ones_like(self.psi)
            gauge_acc = 0.0j * np.ones_like(self.psi)
        return yukawa_acc, gauge_acc

    def gauss(self, x, mu=0, sigma=5):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))



"""coupling_term += -self.i * m * self.psi

                # Zeitschritt
                new_psi = self.psi + self.d["t"] * coupling_term
                # LOGGER.info("new_psi:", new_psi)

                self.psi = new_psi
                
    def _add_g_coupling_to_total(self, total, new_term):
        if self.is_quark:
            for i in range(len(total)):
                for j in range(len(total)):
                    total[i][j] += new_term[i][j]
        else:
            for i in range(len(total)):
                total[i] += new_term[i]
        return total


    def _add_yukawa_coupling_to_total(self, total, new_term):
        if self.is_quark:
            for j in range(len(total)):
                total[j] += new_term[j]
        else:
            total += new_term
        return total                
"""
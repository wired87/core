import itertools

import numpy as np

from qf_sim.physics.quantum_fields.nodes.g.gauge_utils import GaugeUtils
from qf_sim.physics.quantum_fields.qf_core_base.qf_utils import QFUtils
from utils.graph.local_graph_utils import GUtils


class QuadCoupler(GaugeUtils):

    def __init__(self, g_utils, neighbors, parent):
        super().__init__()
        self.parent=parent
        self.g_utils: GUtils = g_utils
        self.qf_utils = QFUtils(g=g_utils)

        self.neighbors: list[tuple] = self._filter_neighbors(
            neighbors
        )
        # receive list neighbors with !=0 field value
        self.active_neighbors: list[tuple] = self._check_active_indizes(
            neighbors=self.neighbors
        )
        self.quads = self._create_quads()

        self.f_abc = getattr(self.parent, "f_abc")

    def main_quad(self):
        attrs = getattr(self.parent, "attrs")
        for t in self.quads:
            self_attrs = t[3][1]
            nrighbor1 = t[2][1]
            nrighbor = t[1][1]
            neighbor2 = t[0][1]
            self.constructor(
                e_charge=attrs["charge"]
            )
            self.contract_quartic_vertex(
                eps1=nrighbor[self._field_value(nrighbor["type"])],
                eps2=neighbor2[self._field_value(neighbor2["type"])],
                eps3=self_attrs[self._field_value(self_attrs["type"])],
                eps4=nrighbor1[self._field_value(self_attrs["type"])],
            )



    def constructor(self, e_charge):
        """
        Berechnet den Quartic Vertex Tensor V[mu, nu, sigma, rho]
        für W- W+ Photon Photon.
        """
        g = np.diag([1, -1, -1, -1])
        V = np.zeros((4, 4, 4, 4), dtype=complex)

        factor = -1j * e_charge ** 2

        for mu in range(4):
            for nu in range(4):
                for sigma in range(4):
                    for rho in range(4):
                        term1 = 2 * g[sigma, rho] * g[mu, nu]
                        term2 = -g[sigma, mu] * g[rho, nu]
                        term3 = -g[sigma, nu] * g[rho, mu]
                        V[mu, nu, sigma, rho] = factor * (term1 + term2 + term3)
        self.V = V

    def contract_quartic_vertex(self, eps1, eps2, eps3, eps4):
        """
        Kontrahiert den Quartic Vertex Tensor mit vier Feldvektoren.

        eps1: Feldvektor 1 (z.B. Photon)
        eps2: Feldvektor 2 (z.B. Photon)
        eps3: Feldvektor 3 (W-)
        eps4: Feldvektor 4 (W+)
        """
        # Apply to j_nu
        j_nu = getattr(self.parent, "j_nu")
        for mu in range(4):
            j_sum = 0
            for nu in range(4):
                for sigma in range(4):
                    for rho in range(4):
                        j_sum += self.V[mu, nu, sigma, rho] * eps1[mu] * eps2[nu] * eps3[sigma] * eps4[rho]

            j_nu[mu] += j_sum
        setattr(self.parent, "j_nu", j_nu)






    def compute_coupling_quad(self, V: list[tuple], f, g=1.0):
        """
        V: list of tuples (4-vector, index)
            [(np.array([4,]), int), ...]
            3 oder 4 Elemente
        f: np.array([8,8,8])
            Strukturkonstanten
        g: float
            Kopplungskonstante
        """
        total = 0.0
        v = [entry[0] for entry in V]
        i = [entry[1] for entry in V]

        # Quad Vertex
        for e in range(8):
            f1 = f[i[0], i[1], e]
            f2 = f[i[2], i[3], e]

            # Skalarprodukte
            s1 = sum(v[0][mu] * v[1][mu] for mu in range(4))
            s2 = sum(v[2][nu] * v[3][nu] for nu in range(4))

            term = f1 * f2 * s1 * s2
            total += term

        total *= -0.25 * g ** 2

        return total

    def _create_quads(self):
        self_nid = getattr(self.parent, "id")
        self_nattrs = getattr(self.parent, "attrs")
        doubles = list(itertools.combinations(self.active_neighbors, 3))
        quads = []
        for item in doubles:
            item = item + (self_nid, self_nattrs)
            quads.append(item)
        return quads

    def _filter_neighbors(self, neighbors):
        """
        Filter neighbors Vertex
        """
        self_ntype = getattr(self.parent, "type")
        node_parent = getattr(self.parent, "parent")
        # todo error
        valid_vertex = self._quad_type_combi(self_ntype)
        fltered_neighbors = []

        for nnid, nattrs in neighbors:
            ntye = nattrs.get("type")
            if ntye in valid_vertex and ntye != self_ntype:
                fltered_neighbors.append(
                    (nnid, nattrs)
                )

       #print("Triple neighbors filtered")
        return fltered_neighbors


















    def _quad_type_combi(self, ntype):
        if ntype.lower() == "z_boson":
            return ["w_plus", "w_minus", "z_boson", "photon"]
        elif ntype.lower() == "photon":
            return ["w_plus", "w_minus", "z_boson", "photon"]
        elif ntype.lower() in ["w_plus", "w_minus"]:
            return ["w_plus", "w_minus", "photon", "z_boson"]
        elif ntype.lower() == "gluon":
            return ["gluon" for _ in range(4)]

    def _check_delete_quads(self, nid):
        gg_coupling_term = getattr(self.parent, "gg_coupling_term")
        if gg_coupling_term <= 0:
            quads = self.g_utils.get_neighbor_list(nid, "QUAD")
            if quads:
                for tid, _ in quads:
                    self.g_utils.delete_node(tid)

    def quad_id_list_powerset(self, neighbors, quad):
        """
        Step 3 Create powersets of possible cons
        """

        id_set = []
        # loop all neghbors k=type, v=node_ids
        for k, v in neighbors.items():
            # appand id
            id_set.append(list(v.keys()))

        needed_len = 4 if quad is True else 3
        if len(id_set) < needed_len:
            return []

        from itertools import product
        if quad is True:
            return list(product(id_set[0], id_set[1], id_set[2], id_set[3]))
        else:
            # tripple
            return list(product(id_set[0], id_set[1], id_set[2]))




"""

self.qf_utils.create_connection(
                    node_data=[
                        {"id": attrs["id"], "type": attrs["type"]}
                        for attrs in nattrs
                    ],
                    coupling_strength=coupling_strength,
                    env_id=getattr(self.parent, "env")["id"],
                    con_type="QUAD",
                    nid=nid
                )
                
                
 def _quads(self, field_value, field_key, self_ntype):
        nid = getattr(self.parent, "id")
        is_gluon = getattr(self.parent, "is_gluon", "")

        ntypes = self._quad_type_combi(self_ntype)
        # Do we have a  Gauge field here?
        if ntypes is not None:
            # combi_ntype: node_id: list[indices] or dict(nattrs)

            # Extract all neighbors with type ntypes_i
            neighbors = self.get_active_neighbors(
                ntypes,
                self_ntype,
                is_gluon
            )

            combis = self.generate_powerset(neighbors, quad=True, is_gluon=is_gluon)
            if not combis:
               #print("Break quad process")
                return

            if is_gluon:
                for c in combis:
                    # [(node_id, idx),... *4]
                    # Extract node attrs and field values
                    node_attrs = {}
                    field_values_extracted = []
                    for j, (nid, vi) in enumerate(c):
                        nattrs = self.g_utils.G.nodes[nid]
                        field_indice = nattrs["G"][vi]
                        node_attrs[nid] = {
                            "field_value": field_indice,
                            "indece": vi,
                            "attrs": nattrs,
                        }
                        field_values_extracted.append((field_indice, vi))

                    # calc coupling between nodes
                    t_coupling = self.compute_coupling(
                        f=self.f_abc,
                        V=field_values_extracted,
                        g=getattr(self.parent, "g")
                    )

                    for k, v in node_attrs.items():
                        # apply increase to != 0 vec indices
                        for i in range(4):
                            field_value_item = v["field_value"][i]
                            if field_value_item > 0:
                                v["field_value"][i] += t_coupling

                        # Update node
                        self.g_utils.update_node(
                            attrs=v["attrs"]
                        )
            else:
                for c in combis:
                    nattrs = []
                    coupling_strength = 1.0
                    for idx in c:
                        if idx != nid:
                            _nattrs = self.g_utils.G.nodes[idx]
                           #print("Nattrs received:", _nattrs)
                            nattrs.append(_nattrs)
                            _ntype = _nattrs.get("type").lower()
                            field_val = _nattrs.get(
                                self._field_value(_ntype)
                            )
                            coupling_strength *= field_val


            self._check_delete_quads(nid)
"""
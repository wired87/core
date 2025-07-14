from itertools import product

import numpy as np

from qf_sim.physics.quantum_fields.nodes.g.coupler.quad import QuadCoupler
from qf_sim.physics.quantum_fields.nodes.g.coupler.triple import TripleCoupler
from qf_sim.physics.quantum_fields.nodes.g.gauge_utils import GaugeUtils

class VertexUtils(TripleCoupler, QuadCoupler):

    def __init__(self, g_utils, neighbors, qf_utils, parent):
        TripleCoupler.__init__(self, g_utils, neighbors, qf_utils, parent)
        QuadCoupler.__init__(self, g_utils, neighbors, parent)

    def _get_vertex_combi(self, quad):
        if quad is True:
            return self._quad_type_combi(self.type)
        else:
            return self._tripple_type_combi(self.type)


class Vertex(VertexUtils):

    def __init__(self, g_utils, neighbors, qf_utils, parent):
        self.active_indices = None
        self.parent=parent
        self.f_abc = getattr(self.parent, "f_abc")
        self.is_gluon = getattr(self.parent, "is_gluon")
        self.type = getattr(self.parent, "type")
        self.neighbors = getattr(self.parent, "neighbors")
        self.field_value = getattr(self.parent, "field_value")
        self.id = getattr(self.parent, "id")
        self.g_utils = getattr(self.parent, "g_utils")

        super().__init__(g_utils, neighbors, qf_utils, parent)





    def update_vertex(
            self,
            j_nu_sum,
            coupling_strength,
            vertex_size:int
    ):
        # erste eigene Gleichung
        total_j_nu = 0.0
        for j in j_nu_sum:
            total_j_nu += np.linalg.norm(j)**2
        energy = (total_j_nu * coupling_strength) / vertex_size
       #print("vertex update:", energy)
        return energy




    def main(self):
        """
        Main entry for vertex defineitions
        Applys the result instant to local j_nu
        """
        # Triplets
        self.triple_main()

        # Quads
        self.main_quad()
        # e_charge, eta_e, w_minus, w_plus, p_photon

       #print("vertex created")



























    def create_possible_vertex_combis(self, active_indices, quad: bool, is_gluon: bool, vertex_size) -> list:
        """
        Erzeugt alle Kombinationen von Nachbarn.
        - Wenn is_gluon=True, wird aus den aktiven Indizes jeder Node kombiniert.
        - Wenn is_gluon=False, werden die Node-IDs kombiniert.

        Args:
            active_indices (dict):
                - Non-Gluon: {type: {node_id: any}}
                - Gluon: {type: {node_id: (indices_list, idx)}}
            quad (bool): True=Quad, False=Tripple
            is_gluon (bool): Steuert Gluon-Logik
        Returns:
            list: Liste der Kombinationen
        """
        id_set = self._filter_node_ids(active_indices)
        result = self._create_powerset(id_set, quad)
       #print("✅ Non-Gluon combinations:", result)

        if is_gluon:
            # Gluon: indices kombinieren
            all_combinations = self.generate_gluon_index_combinations(
                active_indices, vertex_size
            )
           #print("✅ Gluon combinations:", all_combinations)
            return all_combinations

        else:
            # Non-Gluon: Node-IDs kombinieren


            # Create powerset from node id lists
            if len(id_set) < vertex_size:
                return []


            # gluon: [((nid1, lorenz_vec_id),...]
            # !gluon: [(nid1, nid2, nid3),...]
            return result


    def _create_powerset(self, id_set, quad):
        if quad:
            result = list(product(id_set[0], id_set[1], id_set[2], [self.id]))
        else:
            result = list(product(id_set[0], id_set[1], [self.id]))
        return result


    def _filter_node_ids(self, active_indices):
        id_set = []
        for nnode_type, node_ids in active_indices.items():
            # Add node_ids
            node_ids = list(node_ids.keys())
            if not node_ids:
                continue
            id_set.append(node_ids)
        return id_set


    def _extend_powerset(self, vertex_combis:list, active_indices):
        """
        Maske sure

        :return:
        """
        vertex_combi_with_self = []
        for combi in vertex_combis:
            # combi: tuple
            combi = list(combi)
            combi.append(active_indices)
            vertex_combi_with_self.append(combi)

       #print("Vertex combis extendet")



    def get_active_neighbors(self, v_combi) -> dict:
        active_indices = {}

        # 1 COLLECT ALL POSSIBLE NODES FORM NEIGHBORS
        for nnid, nattrs in self.neighbors:
            ntype = nattrs.get("type").lower()

            # Valid Vertex coupling partner (Gauge)?
            if ntype in self.gauge_fields: # gluon | z...

                # Foield value
                nfield_key = self._field_value(ntype)
                nfield_value = nattrs[nfield_key]

                # Get valid coupling neighbors from v_combi
                if ntype in v_combi and ntype != self.type:
                    if ntype not in active_indices:
                        active_indices[ntype] = {}

                    # 2 CHECK ACTIVE INDICES
                    indizes = self._check_active_indizes(nfield_value, nattrs, gluon=self.is_gluon)
                    if indizes is not None:
                        active_indices[ntype][nnid]:list[int] = indizes

        self_active_indizes = self._check_active_indizes(self.field_value)
        if self_active_indizes is None:
           #print("Self inidces is None: return")
            return

        # add self after powerset creation -> self muss eh in jeder combi dabei sein
        #active_indices[self.type.lower()][self.id] = self_active_indizes

       #print("finished get_active_neighbors")
        return active_indices, self_active_indizes







    def compute_coupling_triple(self, values):
        fabc_item = self.f_abc[values[0][0], values[1][0], values[2][0]]

        if fabc_item != 0:
            pass






























"""else:
            for c in combis:
                nattrs = []
                coupling_strength = 1.0
                for idx in c:
                    if idx != self.id:
                        _nattrs = self.g_utils.G.nodes[idx]
                       #print("Nattrs received:", _nattrs)
                        nattrs.append(_nattrs)
                        _ntype = _nattrs.get("type").lower()
                        field_val = _nattrs.get(
                            self._field_value(_ntype)
                        )
                        coupling_strength *= field_val
            for j, (nid, vi) in enumerate(c):
                nattrs = self.g_utils.G.nodes[nid]
                field_indice = nattrs["G"][vi]
                node_attrs[nid] = {
                    "field_value": field_indice,
                    "indece": vi,
                    "attrs": nattrs,
                }
                field_values_extracted.append((field_indice, vi))







    def compute_coupling_quad(self, V:list[tuple], f, g=1.0):
        V1, V2, V3, V4: np.array([4,])
            Die 4-Vektoren der 4 gewählten Komponenten
        f: np.array([8,8,8])
            Strukturkonstanten
        a,b,c,d: int
            Gluon index zu diesen Vektoren
        total = 0.0
        v = [i[0] for i in V]
        i = [j[1] for j in V]

        # Summiere nur e
        for e in range(8):
            f1 = f[i[1], i[2], e]
            f2 = f[i[3], i[4], e]

            # Skalarprodukte aufsummieren
            s1 = sum(v[0][mu] * v[1][mu] for mu in range(4))
            s2 = sum(v[1][nu] * v[3][nu] for nu in range(4))

            term = f1 * f2 * s1 * s2
            total += term

        total *= -0.25 * g ** 2
        return total


        """


"""
def _compute_interaction(self, combis):
    #if self.is_gluon:
    for c in combis:
        # [((nid, nattrs),...),... ]
        node_attrs = {}
        field_values_extracted = []





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
            )"""




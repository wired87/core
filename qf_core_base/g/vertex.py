from itertools import product

import numpy as np

from qf_core_base.g.coupler.quad import QuadCoupler
from qf_core_base.g.coupler.triple import TripleCoupler


class Vertex:

    def __init__(self, parent):
        self.tc = TripleCoupler(parent)
        self.qc = QuadCoupler(parent)
        self.parent=parent


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

    def main(self, neighbors, edges):
        """
        Main entry for vertex defineitions
        Applys the result instant to local j_nu
        """

        # Triplets
        self.tc.triple_main(neighbors, edges)

        # Quads
        self.qc.main_quad(neighbors, edges)
        # e_charge, eta_e, w_minus, w_plus, p_photon
        print("Vertex process finished")


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









    def compute_coupling_triple(self, values):
        fabc_item = self.f_abc[values[0][0], values[1][0], values[2][0]]

        if fabc_item != 0:
            pass





























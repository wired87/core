import itertools
from itertools import product

import numpy as np

from qf_core_base.g.gauge_utils import GaugeUtils
from utils.graph.local_graph_utils import GUtils


class TripleCoupler(GaugeUtils):

    def __init__(self, g_utils, neighbors, qf_utils, parent):
        GaugeUtils.__init__(self)
        self.parent=parent

        self.g_utils: GUtils = g_utils
        self.qf_utils = qf_utils

        self.neighbors:list[tuple] = self._filter_neighbors(
            neighbors
        )
        # receive list neighbors with !=0 field value
        self.active_neighbors:list[tuple] = self._check_active_indizes(
            neighbors=self.neighbors
        )
        self.triplets = self._create_triplets()


    def triple_main(self):
        for t in self.triplets:
            # t already deserialized
            self_attrs= t[2][1]
            nrighbor= t[1][1]
            neighbor2= t[0][1]
            self.calc_triple(
                e_charge=self_attrs["charge"],
                eta_e=self_attrs["g"],
                w_minus=nrighbor[self._field_value(nrighbor["type"])],
                w_plus=neighbor2[self._field_value(neighbor2["type"])],
                p_photon=self_attrs[self._field_value(self_attrs["type"])],
            )

    def calc_triple(self, e_charge, eta_e, w_minus, w_plus, p_photon):
        """
        riccardo calcualtion tripple
        W- W+ Photon Vertex Tensor V[mu, sigma, rho]
        """
        g = np.diag([1, -1, -1, -1])
        V = np.zeros((4, 4, 4), dtype=complex)

        factor = -1j * eta_e * e_charge
        for mu in range(4):
            for sigma in range(4):
                for rho in range(4):
                    term1 = g[sigma, rho] * (w_minus[mu] - w_plus[mu])
                    term2 = g[rho, mu] * (w_plus[sigma] - p_photon[sigma])
                    term3 = g[mu, sigma] * (p_photon[rho] - w_minus[rho])
                    V[mu, sigma, rho] = factor * (term1 + term2 + term3)

        # Apply to j_nu
        j_nu = getattr(self.parent, "j_nu")
        for nu in range(4):
            j_sum = 0
            for mu in range(4):
                for rho in range(4):
                    j_sum += V[mu, nu, rho]
            j_nu[nu] += j_sum
        setattr(self.parent, "j_nu", j_nu)
    def contract_vertex(self, V, eps_photon, eps_W_minus, eps_W_plus):
        """
        Kontrahiert den Tensor mit drei Polarisationen
        """
        result = 0.0 + 0.0j
        for mu in range(4):
            for sigma in range(4):
                for rho in range(4):
                    result += V[mu, sigma, rho] * eps_photon[mu] * eps_W_minus[sigma] * eps_W_plus[rho]
        return result



    def _create_triplets(self):
        self_nid = getattr(self.parent, "id")
        self_nattrs = getattr(self.parent, "attrs")
        doubles = list(itertools.combinations(self.active_neighbors, 2))
        triplets = []
        for item in doubles:
            item = item + (self_nid, self_nattrs)
            triplets.append(item)
        return triplets


    def _filter_neighbors(self, neighbors):
        """
        Filter neighbors Vertex
        """
        self_ntype = getattr(self.parent, "type")
        node_parent = getattr(self.parent, "parent")
        valid_vertex = self._tripple_type_combi(self_ntype)
        fltered_neighbors = []

        for nnid, nattrs in neighbors:
            ntye = nattrs.get("type")
            if ntye in valid_vertex and ntye != self_ntype:
                fltered_neighbors.append(
                    (nnid, nattrs)
                )

       #print("Triple neighbors filtered")
        return fltered_neighbors












    def _create_tripple_connection_from_ps(self, combis, f_abc, field_key, nid, self_field_value):
        # Check for a fabc match
        attrs = getattr(self.parent, "attrs", {})
        for combi in combis:
            # bsp. combi: {nid: (v1, vi), nid2: ...}
            values = list(combi.values())
            fabc_item = f_abc[values[0][0], values[1][0], values[2][0]]

            if fabc_item != 0:
                ids = list(combi.keys())
                coupling_strength = fabc_item
                node_data_list = [attrs]

                for i, _nid in enumerate(ids):
                    if _nid != nid:
                        nattrs = self.g_utils.G.nodes[_nid]

                        gluon_index = combi[_nid][0]

                        n_field_value = nattrs[field_key][gluon_index]
                        coupling_strength *= np.sum(n_field_value)
                        node_data_list.append(nattrs)
                    else:
                        coupling_strength *= np.sum(self_field_value)

                # apply coupling to color_indices
                for _nid in ids:
                    gluon_index = combi[_nid][0]
                    g_component_index = combi[_nid][1]
                    if _nid != nid:
                        # Get n field value
                        nattrs = self.g_utils.G.nodes[_nid]
                        n_field_value = nattrs[field_key]

                        # update singel float in field value
                        n_field_value[gluon_index, g_component_index] = coupling_strength

                        self.g_utils.update_node(
                            getattr(self.parent, "attrs", {}).update(
                                {field_key: self_field_value}
                            )
                        )

                    else:
                        # update singel float in field value ad update attrs
                        self_field_value[gluon_index, g_component_index] = coupling_strength
                        setattr(self, field_key, self_field_value)
                        return self_field_value

                """self.qf_utils.create_connection(
                    node_data=[
                        {
                            "id": attrs["id"],
                            "type": attrs["type"]
                        } for attrs in node_data_list
                    ],
                    coupling_strength=coupling_strength,
                    env_id=getattr(self, "env")["id"],
                    con_type="TRIPPLE",
                    nid=nid
                )"""
       #print("Tripple connection process finshed")




    def _tripple_type_combi(self, ntype):
        if ntype.lower() == "z_boson":
            return ["w_plus", "w_minus", "z_boson"]
        elif ntype.lower() == "photon":
            return ["w_plus", "w_minus", "photon"]
        elif ntype.lower() == "w_plus" or ntype.lower() in "w_minus":
            return [["w_plus", "w_minus", "photon"], ["w_plus", "w_minus", "z_boson"]]
        elif ntype.lower() == "gluon":
            return ["gluon" for _ in range(3)]



    def _check_delete_triples(self, nid):
        gg_coupling_term = getattr(self.parent, "gg_coupling_term")
        if gg_coupling_term <= 0:
            triples = self.g_utils.get_neighbor_list(nid, "TRIPPLE")
            if len(triples):
                for tid, _ in triples:
                    self.g_utils.delete_node(tid)










"""




    def generate_combinations(self, input_dict) -> list[dict]:
        Erzeugt alle Kombinationen der Werte aus den 3 Listen jedes Dict-Keys.
        Args:
            input_dict (dict): {key1: [(v1, v1i), (v2, v2i),...]
        Returns:
            list[dict]: Jede Kombination als dict {key1: val1, key2: val2, key3: val3}
        keys = list(input_dict.keys())
        # just take actual gluon component index without actual element (float) index
        value_lists = [input_dict[k][0] for k in keys]
        all_combinations = []

        for combo in product(*value_lists):
            # apply index hereas tuple
            combo_dict = {k: (v, input_dict[k][1]) for k, v in zip(keys, combo)}
            all_combinations.append(combo_dict)
       #print("all_combinations", all_combinations)
        return all_combinations

    def _triples(self, field_value, field_key, ntype):
        nid = getattr(self.parent, "id")

        ntypes = self._tripple_type_combi(ntype)

        is_gluon = "gluon" in ntypes
        # Sort all enighbors of tripplet combi types in a new dict

        all_neighbor_indices={}

        for nnid, nattrs in self.neighbors:
            nntype = nattrs.get("type").lower()
            if nntype in self.gauge_fields:
               #print(">>>nattrs", nattrs)

                nfield_key = self._field_value(nntype)

                nfield_value = nattrs[nfield_key]
                # Extract active field values
                if nntype in ntypes:
                    if nntype not in all_neighbor_indices:
                        all_neighbor_indices[nntype] = {}
                        all_neighbor_indices[nntype][nnid]:list or dict = None
                    # get either active indices of a gluon as list or
                    # nattrs of neighbors
                    indizes = self._check_active_indizes(nfield_value, nattrs, gluon=is_gluon)
                    if indizes is not None:
                        all_neighbor_indices[nntype][nnid]: list or dict = indizes

        combis = self.create_combi(
            nid,
            all_neighbor_indices,
            field_value
        )

        if combis is None:
           #print("Break triple process")
            return

        if is_gluon is True:
            f_abc = getattr(self.parent, "f_abc")
            # TRIPPLE
            self._create_tripple_connection_from_ps(
                combis, f_abc, field_key, nid, self_field_value=field_value
            )
        else:
            for c in combis:
                nattrs = []
                coupling_strength = 1.0
                for idx in c:
                    if idx != nid:
                        _nattrs = self.g_utils.G.nodes[idx]
                        nattrs.append(_nattrs)
                        _ntype = _nattrs.get("type")
                        field_value = _nattrs.get(_ntype[0].upper())
                        coupling_strength *= field_value

                self.qf_utils.create_connection(
                    node_data=[
                        {
                            "id": attrs["id"],
                            "type": attrs["type"]
                        } for attrs in nattrs
                    ],
                    coupling_strength=coupling_strength,
                    env_id=getattr(self.parent, "env")["id"],
                    con_type="TRIPPLE",
                    nid=nid
                )

        # DELETE
        self._check_delete_triples(nid)


"""




















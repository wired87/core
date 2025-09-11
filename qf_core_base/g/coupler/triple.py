import itertools

import numpy as np

from qf_core_base.g.gauge_utils import GaugeUtils

class TripleCoupler(GaugeUtils):

    def __init__(
            self,
            parent,
    ):
        GaugeUtils.__init__(self)
        self.parent = parent

        # todo
        self.triplets=None

    def triple_main(self, neighbors, edges):
        self.edges=edges
        self.neighbors=neighbors

        if self.triplets is None:
            self.self_nid = getattr(self.parent, "id")
            self.self_nattrs = getattr(self.parent, "attrs")
            self_ntype = getattr(self.parent, "type")

            valid_vertex: list[str or list] = self._tripple_type_combi(self_ntype)
            #print("valid_vertex", valid_vertex)

            # w+- can have partnerships with z and p
            if not isinstance(valid_vertex[0], list):
                valid_vertex = [valid_vertex]

            for combi_list in valid_vertex:
                triplets = self.ceate_powerset(
                    self_ntype,
                    self.self_nid,
                    combi_list,
                    self.neighbors,
                    self.self_nattrs,
                    quad=False
                )
                self.work_tripplets(triplets)


    def work_tripplets(self, triplets):
        for t in triplets:
            # t already deserialized
            #print(f"triplet t:")
            #pprint.pp(t)

            self_attrs= t[2]
            nrighbor= t[1]
            neighbor2= t[0]
            self.calc_triple(
                self_attrs,
                nrighbor,
                neighbor2,
                e_charge=self_attrs["charge"],
                eta_e=self_attrs["g"],
            )

    def calc_triple(self, self_attrs, n1, n2, e_charge, eta_e):
        """
        riccardo calcualtion tripple
        W- W+ Photon Vertex Tensor V[mu, sigma, rho]
        """
        self.self_nattrs = getattr(self.parent, "attrs")
        g = np.diag([1, -1, -1, -1])
        V = np.zeros((4, 4, 4), dtype=complex)

        w_minus = n2[self._field_value(n2["type"])]
        w_plus = n1[self._field_value(n1["type"])]
        p_photon = self_attrs[self._field_value(self_attrs["type"])]

        if np.sum(w_plus) > 0 and np.sum(w_minus) > 0 and np.sum(p_photon) > 0:
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
            print(f"f^{{abc}} V_{{μνρ}}(p₁, p₂, p₃): {j_nu}")
            self.update_edges(
                neighbors=[
                    n1, n2
                ],
                j_nu=j_nu
            )


    def update_edges(
            self,
            neighbors,
            j_nu,
    ):
        # Save j_nu in edges
        ids = [n["id"] for n in neighbors]
        ids = [self.self_nid, *ids]
        identifier = f"_".join(ids)

        # create edges between all
        for id1, id2 in itertools.combinations(ids, 2):
            for eid, eattrs in self.edges.items():
                if id1 in eid and id2 in eid:
                    eattrs.update(
                        {"cons": {identifier: j_nu}}
                    )

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

                        nattrs = self.get_node(_nid)

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
                        nattrs = self.get_node(_nid)
                        n_field_value = nattrs[field_key]

                        # update singel float in field value
                        n_field_value[gluon_index, g_component_index] = coupling_strength

                        self.update_node(field_key, self_field_value)
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
        print("ntype _tripple_type_combi", ntype)
        if ntype.lower() == "z_boson":
            return ["w_plus", "w_minus", "z_boson"]
        elif ntype.lower() == "photon":
            return ["w_plus", "w_minus", "photon"]
        elif ntype.lower() == "w_plus" or ntype.lower() in "w_minus":
            return [["w_plus", "w_minus", "photon"], ["w_plus", "w_minus", "z_boson"]]
        elif ntype.lower() == "gluon":
            return ["gluon" for _ in range(3)]














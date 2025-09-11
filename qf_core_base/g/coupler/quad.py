import itertools
import numpy as np
from qf_core_base.g.gauge_utils import GaugeUtils



class QuadCoupler(GaugeUtils):

    def __init__(
            self,
            parent
    ):
        super().__init__()
        self.parent=parent

        self.quads = None


    def main_quad(self, neighbors, edges):
        self.edges = edges
        self.neighbors = neighbors

        if self.quads is None:
            self.quads = self._create_powerset()

        attrs = getattr(self.parent, "attrs")
        for t in self.quads:
            nrighbor1 = t[2]
            nrighbor = t[1]
            neighbor2 = t[0]

            self.constructor(
                e_charge=attrs["charge"]
            )

            self.contract_quartic_vertex(
                nrighbor,
                nrighbor1,
                neighbor2,
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
        print(f"V_{{μνσρ}}^{{WWγγ}}: {V}")

    def contract_quartic_vertex(
            self,
            nrighbor,
            nrighbor1,
            neighbor2,
    ):
        """
        Kontrahiert den Quartic Vertex Tensor mit vier Feldvektoren.
        eps1: Feldvektor 1 (z.B. Photon)
        eps2: Feldvektor 2 (z.B. Photon)
        eps3: Feldvektor 3 (W-)
        eps4: Feldvektor 4 (W+)
        """
        eps1 = nrighbor[self._field_value(nrighbor["type"])]
        eps2 = neighbor2[self._field_value(neighbor2["type"])]
        eps3 = self.nattrs[self._field_value(self.nattrs["type"])]
        eps4 = nrighbor1[self._field_value(nrighbor1["type"])]
        if np.sum(eps1) > 0 and np.sum(eps2) > 0 and np.sum(eps3) > 0 and np.sum(eps4) > 0:
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

            print(f"f^{{abe}}f^{{cde}} V_{{μνρσ}}(p₁, p₂, p₃, p₄): {j_nu}")

            self.update_edges(
                neighbors=[
                    nrighbor,
                    nrighbor1,
                    neighbor2,
                ],
                j_nu=j_nu
            )

    def update_edges(
            self,
            neighbors,
            j_nu
    ):
        # Save j_nu in edges
        ids = [n["id"] for n in neighbors]
        ids = [self.nid, *ids]
        identifier = f"_".join(ids)

        # create edges between all
        for id1, id2 in itertools.combinations(ids, 2):
            for eid, eattrs in self.edges.items():
                if id1 in eid and id2 in eid:
                    eattrs.update(
                        {"cons": {identifier: j_nu}}
                    )


    def _create_powerset(self):
        """
        Filter neighbors Vertex
        """
        self_ntype = getattr(self.parent, "type")
        self.f_abc = getattr(self.parent, "f_abc")
        self.nid = getattr(self.parent, "id")
        self.nattrs = getattr(self.parent, "attrs")

        valid_vertex:list[str] = self._quad_type_combi(self_ntype)

        return self.ceate_powerset(
            self_ntype,
            self.nid,
            valid_vertex,
            self.neighbors,
            self.nattrs
        )

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
        print("ntype _quad_type_combi", ntype)
        if ntype.lower() == "z_boson":
            return ["w_plus", "w_minus", "z_boson", "photon"]
        elif ntype.lower() == "photon":
            return ["w_plus", "w_minus", "z_boson", "photon"]
        elif ntype.lower() in ["w_plus", "w_minus"]:
            return ["w_plus", "w_minus", "photon", "z_boson"]
        elif ntype.lower() == "gluon":
            return ["gluon" for _ in range(4)]

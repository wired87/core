
import numpy as np

from qf_core_base.calculator.calculator import Calculator
from qf_core_base.higgs.phi_utils import HiggsUtils
from qf_core_base.qf_utils.all_subs import FERMIONS, G_FIELDS
from qf_core_base.qf_utils.field_utils import FieldUtils
from qf_core_base.qf_utils.qf_utils import QFUtils

from qf_core_base.symmetry_goups.main import SymMain
from utils._np.serialize_complex import check_serialize_dict


class HiggsBase(FieldUtils, HiggsUtils):
    """
    Das Higgsfeld hat nach SSB einen konstanten Vakuumwert
    𝑣
    ≈
    Die Yukawa-Kopplung
    Selbstwechselwirkungsterme -> Higgs higgs interkation (unterschliech von dmu_phi(dphi = kinetische energeie)
    Masse ergibt sich statisch:
    ein punkt kann mehrere higgs bosonen erzeugen welche wechslwirken

    Higgs muss nciht getriggrt werden oder sonst was.
    Erreicht eine welle ein femrion punkt inm raum gibt H
    ihm masse (wie handshake)
    """

    def __init__(
            self,
            g,
            d,
            neighbors_pm,
            time,
            attrs,
            env,
            **args,
    ):
        super().__init__()
        self.attrs = self.restore_selfdict(data=attrs)
        # make class attrs
        # LOGGER.info("init FermionBase")
        for k, v in self.attrs.items():
            setattr(self, k, v)
            # LOGGER.info(f"{k}:{v}")
        self.energy = None
        self.d_phi = None

        self.g = g
        self.env = env
        self.d = d  # distance
        self.neighbors_pm = neighbors_pm
        self.parent = self.attrs["parent"][0].lower()

        self._laplacian_h()
        self._lambda_H()
        self._higgs_potential_derivative()

        self.attr_keys = [k for k in attrs.keys()]


        self.symmetry_group_class = SymMain(groups=getattr(self, "_symmetry_groups", [])[0])
        self.calculator = Calculator(g)
        self.qf_utils = QFUtils(g)

        self.neighbors = self.g.get_neighbor_list(
            getattr(self, "id"),
            self.all_sub_fields
        )

    def main(self):
        # Set prev value
        #print("Update phi")
        h = getattr(self, "h")
        nid = getattr(self, "id")

        if self.h_prev is None:
            self.phi_prev = h

        self.h_prev = h
        self._d_phi()
        self._h()
        self._phi()
        self._energy_density()
        self._coupling(nid)

        new_dict = check_serialize_dict(
            self.__dict__,
            self.attr_keys
        )



        self.g.update_node(new_dict)
        # LOGGER.info(f"finiehsed update of {self.id}:")
        #print(f"Update for {nid} finished")


    def _phi(self):
        h = getattr(self, "h")
        vev = getattr(self, "vev")
        #print("h", h)
        #print("vev", vev)
        new_phi = (1/np.sqrt(2)) * np.array([0, vev + h])
        setattr(self, "phi", new_phi)
        #print("_phi set")

    def _coupling(self, nid):
        h = getattr(self, "h")
        for n in self.neighbors:
            nnid = n[0]
            nattrs = n[1]

            ntype = nattrs.get("type")
            phi = getattr(self, "phi")

            coupling_term = None
            if ntype in FERMIONS:
                sub_type = nattrs.get("sub_type", "")
                if sub_type.lower() == "item":
                    coupling_term = self.symmetry_group_class.sym_classes.yukawa_term(
                        **self.attrs,
                        **nattrs,
                    )

            elif ntype in G_FIELDS:
                g = nattrs.get("g")
                field_value = nattrs.get(self._field_value(ntype))
                coupling_term = self.hg_coupling(
                    field_value,
                    g,
                    phi,
                )

            if coupling_term is not None:
                self.g.update_edge(
                    src=nid,
                    trgt=nnid,
                    rels=["intern_coupling", "extern_coupling"],
                    attrs=self.attrs.update(
                        {"coupling_term": coupling_term}
                    )
                )

                h += self.d["t"] * coupling_term.real
                setattr(self, "h", h)
                # LOGGER.info(f"Φ = {self.h}")



    def _d_phi(self):
        # LOGGER.info("neighbors_pm", self.neighbors_pm)
        # LOGGER.info("self.h", self.h)

        phi_t = self.calculator.cutils._dmuX(
            self.h, self.phi_prev, self.d["t"], time=True
        )  # dt = timestep
        dphi = [
            self.init_phi(h=phi_t, serialize=False)
        ]
        # # LOGGER.info("neighbors", self.neighbors_pm)
        for i, (key, pm) in enumerate(self.neighbors_pm.items()):  # x,y,z
            # LOGGER.info("Dphi run", i)
            plus_id = pm[0]
            minus_id = pm[1]

            neighhbor_plus = self.g.get_single_neighbor_nx(plus_id, self.type.upper())
            neighhbor_minus = self.g.get_single_neighbor_nx(minus_id, self.type.upper())

            phi_plus = neighhbor_plus[1]["h"]
            # LOGGER.info(f"nplus {minus_id}-> {neighhbor_plus[1]}:{phi_plus}")

            phi_minus = neighhbor_minus[1]["h"]
            # LOGGER.info(f"nminus {plus_id}-> {neighhbor_minus[1]}:{phi_minus}")

            if phi_plus is None:
                phi_plus = self.h
            if phi_minus is None:
                phi_minus = self.h

            dphi_x_h = self.calculator.cutils._dmuX(
                phi_plus, phi_minus, self.d[key]
            )

            # convert to doublet
            d_phi_X = self.init_phi(h=dphi_x_h, serialize=False)

            ## LOGGER.info(f"dphi{i}", d_phi_X)
            dphi.append(d_phi_X)  # -> alle koords
        #print("Finished dphi", dphi)
        setattr(self, "d_phi", np.array(dphi, dtype=complex))




    def _mass_term(self):
        return -self.mass ** 2 * self.h

    def _h(self):
        h = getattr(self, "h")
        h_prev = getattr(self, "h_prev")
        laplacian_h = getattr(self, "laplacian_h")
        dV_dh = getattr(self, "dV_dh")

        # Klein-Gordon-Update (explizit)
        h_next = 2 * h - h_prev + self.d["t"] ** 2 * (laplacian_h + self._mass_term() - dV_dh)
        # LOGGER.info("h set", h_next)
        setattr(self, "h", h_next)

    def _higgs_potential_derivative(self):
        """
        ∂V/∂h = λ_H * h * (h + v)^2
        Beschreibung: Lokale Rückstellkraft auf das Higgsfeld h
        nach SSB.
        """
        h = getattr(self, "h")
        vev = getattr(self, "vev")
        lambda_H = getattr(self, "lambda_h")
        mu = self.compute_mu(vev, lambda_H)
        dV_dh = -mu ** 2 * (vev + h) + lambda_H * (vev + h) ** 3
        #dV_dh = self.lambda_h * h * (h + vev) ** 2
        setattr(self, "dV_dh", dV_dh)

    def compute_mu(self, vev, lambda_h):
        """
        Berechnet μ aus VEV und λ_H.

        Args:
            vev (float): VEV des Higgsfelds (z.B. 246 GeV)
            lambda_h (float): Quartische Kopplung λ_H

        Returns:
            mu (float): μ in denselben Einheiten wie vev
        """
        import numpy as np
        mu = vev * np.sqrt(lambda_h)
        return mu








    def _lambda_H(self):
        """
        Higgs-Selbstkopplung
        λh bestimmt, wie steil der Mexican Hat ist
        λ_H = m_H² / (2 * v²)
        """
        m = getattr(self, "mass", [])
        vev = getattr(self, "vev", [])
        lambda_h = (m ** 2) / (2 * vev ** 2)
        # LOGGER.info("lambda_h set", lambda_h)
        setattr(self,"lambda_h",lambda_h)

    def _energy_density(self):
        """
        Lokale Energiedichte des Feldpunktes:
        E = 0.5*(∂_t h)^2 + 0.5*(∇h)^2 + V(h)
        Hamiltonian = k*(L0^2/2 + 2*l1*(l1*cos(q1) - L0)*cos(q1)) - g*(l1*m1*sin(q1) + l2*m2*cos(q2)) + (m3*a*a + m1*l1*l1*p2*p2 + 4*b*(b*(m2 + m3) + m3*a*cos(q2)))/(2*l1^2*l2^2*m3*(m1 + 4*(m2 + m3*sin(q2)^2)*sin(q1)^2))
        """
        kinetic = 0.5 * (self.d_phi[0]) ** 2  # Energie, die durch zeitliche Änderungen entsteht
        gradient = 0.5 * sum([c ** 2 for c in self.d_phi[1:]])
        potential = self._higgs_potential()
        # LOGGER.info("kinetic", kinetic)
        # LOGGER.info("gradient", gradient)
        s_gradient = np.abs(gradient)
        self.energy = np.array(kinetic + s_gradient + potential).tolist()
        # LOGGER.info("self.energy updated", self.energy, type(self.energy))

    def _higgs_potential(self):
        """
        V(h) = 0.5 m^2 h^2 + λ v h^3 + 0.25 λ h^4
        """
        m2 = self.mass ** 2
        v = self.vev
        l = self.lambda_h
        h = self.h
        h_potential = (
                0.5 * m2 * h ** 2
                + l * v * h ** 3
                + 0.25 * l * h ** 4
        )
        # LOGGER.info("h_potential", h_potential)
        return h_potential

    def _laplacian_h(self):
        h = getattr(self, "h", None)
        laplacian_h = 0
        for coord, (plus, minus) in self.neighbors_pm.items():
            p_phi = self.g.get_single_neighbor_nx(plus, "PHI")[1]["h"]
            m_phi = self.g.get_single_neighbor_nx(minus, "PHI")[1]["h"]
            laplacian_h += (p_phi + m_phi - 2 * h) / self.env["d"][coord] ** 2
        # LOGGER.info("laplacian set", laplacian_h)
        setattr(self, "laplacian_h", laplacian_h)

import numpy as np

from qf_core_base.higgs.phi_utils import HiggsUtils
from qf_core_base.qf_utils.field_utils import FieldUtils
from _ray_core.ray_validator import RayValidator
from utils._np.serialize_complex import check_serialize_dict

class HiggsBase(
    FieldUtils,
    HiggsUtils,
    RayValidator
):
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
            d,
            attr_keys:list,
            g=None,  # just if non ray
            host=None,  # exists if
    ):
        RayValidator.__init__(self, g_utils=g, host=host)
        HiggsUtils.__init__(self)
        FieldUtils.__init__(self)
        self.symbol = "Φ"
        if host is None:
            self.g = g
            self.host = None
        else:
            self.g = None
            self.host = host

        self.d = d  # distance
        self.attr_keys = attr_keys


    def main(
            self,
            env,
            attrs,
            all_subs:dict,
            neighbor_pm_val_same_type,
            **kwargs,
    ):
        # Args
        self.env=env
        self.all_subs=all_subs
        self.neighbor_pm_val_same_type = neighbor_pm_val_same_type

        # attrs
        self.attrs = self.restore_selfdict(data=attrs)
        for k, v in self.attrs.items():
            setattr(self, k, v)

        # vars
        if self.attrs.get("h_prev") is None:
            self.h_prev = getattr(self, "h")
            self.attrs["h_prev"] = self.h_prev

        self._laplacian_h()
        self._lambda_H()
        self._higgs_potential_derivative()
        self._d_phi()
        self._h()
        self._phi()
        self._energy_density()

        new_dict = check_serialize_dict(
            self.__dict__,
            self.attr_keys
        )
        return new_dict


    def _d_phi(self):
        #print("self.neighbor_pm_val_same_type", self.neighbor_pm_val_same_type)
        #print("self.attrs", self.attrs)
        self.d_phi = self.call(
            method_name="_dX",
            attrs=self.attrs,
            d=self.d,
            neighbor_pm_val_same_type=self.neighbor_pm_val_same_type,
            field_key="h"
        )
        #print(f"∂μh(x):{self.d_phi}")



    def _phi(self):
        try:
            h = getattr(self, "h")
            vev = getattr(self, "vev")
            #print("h", h)
            #print("vev", vev)
            new_phi = (1/np.sqrt(2)) * np.array([0, vev + h])
            setattr(self, "phi", new_phi)
            #print(f"{self.symbol}: {new_phi}")
        except Exception as e:
            print(f"Error phi: {e}")



    def _mass_term(self):
        return -self.mass ** 2 * self.h

    def _h(self):
        try:
            h = getattr(self, "h")
            h_prev = getattr(self, "h_prev")
            laplacian_h = getattr(self, "laplacian_h")
            dV_dh = getattr(self, "dV_dh")

            # Klein-Gordon-Update (explizit)
            h_next = 2 * h - h_prev + self.d["t"] ** 2 * (laplacian_h + self._mass_term() - dV_dh)
            # LOGGER.info("h set", h_next)
            setattr(self, "h", h_next)
            #print(f"h(x): {h_next}")
        except Exception as e:
            print(f"Error _h: {e}")
    def _higgs_potential_derivative(self):
        """
        ∂V/∂h = λ_H * h * (h + v)^2
        Beschreibung: Lokale Rückstellkraft auf das Higgsfeld h
        nach SSB.
        """
        try:
            h = getattr(self, "h")
            vev = getattr(self, "vev")
            lambda_H = getattr(self, "lambda_h")
            mu = self.compute_mu(vev, lambda_H)
            dV_dh = -mu ** 2 * (vev + h) + lambda_H * (vev + h) ** 3
            #dV_dh = self.lambda_h * h * (h + vev) ** 2
            setattr(self, "dV_dh", dV_dh)
            #print(f"∂V/∂h = {dV_dh}")
        except Exception as e:
            print(f"Error _higgs_potential_derivative: {e}")

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
        try:
            m = getattr(self, "mass", [])
            vev = getattr(self, "vev", [])
            lambda_h = (m ** 2) / (2 * vev ** 2)
            # LOGGER.info("lambda_h set", lambda_h)
            setattr(self, "lambda_h", lambda_h)
            #print(f"λ_H = {lambda_h}")
        except Exception as e:
            print(f"Error _lambda_H: {e}")

    def _energy_density(self):
        """
        Lokale Energiedichte des Feldpunktes:
        E = 0.5*(∂_t h)^2 + 0.5*(∇h)^2 + V(h)
        Hamiltonian = k*(L0^2/2 + 2*l1*(l1*cos(q1) - L0)*cos(q1)) - g*(l1*m1*sin(q1) + l2*m2*cos(q2)) + (m3*a*a + m1*l1*l1*p2*p2 + 4*b*(b*(m2 + m3) + m3*a*cos(q2)))/(2*l1^2*l2^2*m3*(m1 + 4*(m2 + m3*sin(q2)^2)*sin(q1)^2))
        """
        try:
            kinetic = 0.5 * (self.d_phi[0]) ** 2  # Energie, die durch zeitliche Änderungen entsteht
            gradient = 0.5 * sum([c ** 2 for c in self.d_phi[1:]])
            potential = self._higgs_potential()
            # LOGGER.info("kinetic", kinetic)
            # LOGGER.info("gradient", gradient)
            s_gradient = np.abs(gradient)
            self.energy = np.array(kinetic + s_gradient + potential).tolist()
            # LOGGER.info("self.energy updated", self.energy, type(self.energy))
            #print(f"E = {self.energy}")
        except Exception as e:
            print(f"Error energy density: {e}")

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
        try:
            h = getattr(self, "h", None)
            laplacian_h = 0
            for key, item in self.neighbor_pm_val_same_type.items():
                if key in ["x", "y", "z"]:
                    item:[tuple, tuple]
                    p_item = item[0]
                    m_item = item[1]

                    # extract value
                    val_plus = p_item[1]
                    val_minus = m_item[1]

                    laplacian_h += (val_plus + val_minus - 2 * h) / self.env["d"][key] ** 2
            # LOGGER.info("laplacian set", laplacian_h)
            setattr(self, "laplacian_h", laplacian_h)
        except Exception as e:
            print(f"Error _laplacian_h: {e}")




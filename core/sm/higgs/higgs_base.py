import numpy as np
from core.sm.higgs.phi_utils import HiggsUtils
from qf_utils.field_utils import FieldUtils
from core._ray_core.utils.ray_validator import RayValidator

class HiggsBase(
    FieldUtils,
    HiggsUtils,
    RayValidator
):
    """
    Das Higgsfeld hat nach SSB einen konstanten Vakuumwert
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
            env,
            #g=None,  # just if non ray
    ):
        #RayValidator.__init__(self, g_utils=g, host=None)
        HiggsUtils.__init__(self)
        FieldUtils.__init__(self)

    def main(
            self,
            attrs,
            neighbor_pm_val_same_type,
    ):
        self.laplacian_h(attrs)
        self.lambda_H(attrs)
        self.higgs_potential_derivative(attrs)
        self.d_phi(attrs, neighbor_pm_val_same_type)
        self.phi(attrs)
        self.h(attrs)
        self.energy_density(attrs)



    def d_phi(self, attrs, neighbor_pm_val_same_type, env):
        d_phi = self._dX(
            attrs=attrs,
            d=env["d"],
            neighbor_pm_val_same_type=neighbor_pm_val_same_type,
            field_key="h"
        )
        print(f"∂μh(x):{d_phi}")
        #attrs["d_phi"] = d_phi
        return d_phi


    def phi(self, attrs):
        h = attrs["h"]
        vev = attrs["vev"]
        symbol = "Φ"
        phi = (1/np.sqrt(2)) * np.array([0, vev + h])
        print(f"{symbol}: {phi}")
        #attrs[""] = new_phi
        return phi

    def mass_term(self, h, mass):
        return -mass ** 2 * h

    def h(self, attrs, env):
        # Klein-Gordon-Update (explizit)
        h=attrs["h"]
        h_prev= attrs["h_prev"]
        laplacian_h=attrs["laplacian_h"]
        dV_dh=attrs["dV_dh"]
        d=env["d"]
        mass_term=attrs["mass_term"]
        h = 2 * h - h_prev + d["t"] ** 2 * (laplacian_h + mass_term - dV_dh)
        print(f"h(x): {h}")
        #attrs["h"] = h
        return h

    def higgs_potential_derivative(
            self,
            attrs
    ):
        vev=attrs["vev"]
        lambda_h=attrs["lambda_h"]
        h=attrs["h"]
        mu = vev * np.sqrt(lambda_h)
        dV_dh = -mu ** 2 * (vev + h) + lambda_h * (vev + h) ** 3
        print(f"∂V/∂h = {dV_dh}")
        #attrs["dV_dh"] = dV_dh
        return dV_dh




    def lambda_H(self, attrs):
        """
        Higgs-Selbstkopplung
        λh bestimmt, wie steil der Mexican Hat ist
        λ_H = m_H² / (2 * v²)
        """
        mass=attrs["mass"]
        vev=attrs["vev"]
        lambda_h = (mass ** 2) / (2 * vev ** 2)
        print(f"λ_H = {lambda_h}")
        attrs["lambda_h"] = lambda_h

    def energy_density(self, attrs):
        """
        Lokale Energiedichte des Feldpunktes:
        E = 0.5*(∂_t h)^2 + 0.5*(∇h)^2 + V(h)
        Hamiltonian = k*(L0^2/2 + 2*l1*(l1*cos(q1) - L0)*cos(q1)) - g*(l1*m1*sin(q1) + l2*m2*cos(q2)) + (m3*a*a + m1*l1*l1*p2*p2 + 4*b*(b*(m2 + m3) + m3*a*cos(q2)))/(2*l1^2*l2^2*m3*(m1 + 4*(m2 + m3*sin(q2)^2)*sin(q1)^2))
        """
        higgs_potential = attrs["higgs_potential"]
        d_phi = attrs["d_phi"]

        kinetic = 0.5 * (d_phi[0]) ** 2  # Energie, die durch zeitliche Änderungen entsteht
        gradient = 0.5 * sum([c ** 2 for c in d_phi[1:]])

        s_gradient = np.abs(gradient)

        energy = np.array(kinetic + s_gradient + higgs_potential).tolist()
        print(f"E = {energy}")
        #attrs["energy"]=energy
        return energy

    def higgs_potential(self, attrs):
        """
        V(h) = 0.5 m^2 h^2 + λ v h^3 + 0.25 λ h^4
        """
        m2 = attrs["mass"] ** 2
        v = attrs["vev"]
        l = attrs["lambda_h"]
        h = attrs["h"]
        h_potential = (
                0.5 * m2 * h ** 2
                + l * v * h ** 3
                + 0.25 * l * h ** 4
        )
        return h_potential

    def laplacian_h(self, attrs, env, neighbor_pm_val_same_type):
        h = attrs["h"]
        laplacian_h = 0
        for key, item in neighbor_pm_val_same_type.items():
            item:[tuple, tuple]
            p_item = item[0]
            m_item = item[1]

            # extract value
            val_plus = p_item[1]
            val_minus = m_item[1]

            laplacian_h += (val_plus + val_minus - 2 * h) / env["d"][key] ** 2
        attrs["laplacian_h"] = laplacian_h
"""

        # Args
        self.all_subs=all_subs
        self.neighbor_pm_val_same_type = neighbor_pm_val_same_type

        # attrs
        self.attrs = self.restore_selfdict(admin_data=attrs)
        for k, v in self.attrs.items():
            setattr(self, k, v)

        # vars
        if self.attrs.get("h_prev") is None:
            self.h_prev = getattr(self, "h")
            self.attrs["h_prev"] = self.h_prev
"""



class HiggsBase(
    FieldUtils,
    HiggsUtils,
):
    """
    Das Higgsfeld hat nach SSB einen konstanten Vakuumwert
    Die Yukawa-Kopplung
    Selbstwechselwirkungsterme -> Higgs higgs interkation (unterschliech von dmu_phi(dphi = kinetische energeie)
    Masse ergibt sich statisch:
    ein punkt kann mehrere higgs bosonen erzeugen welche wechslwirken

    Higgs muss nciht getriggrt werden oder sonst was.
    Erreicht eine welle ein femrion punkt inm raum gibt H
    ihm masse (wie handshake)
    """

    def __init__(self):
        HiggsUtils.__init__(self)
        FieldUtils.__init__(self)
        self.d = self.env["d"]  # distance
        self.symbol = "Φ"

    def main(
            self,
            attrs,
            neighbor_pm_val_same_type,
    ):
        self.laplacian_h(attrs, neighbor_pm_val_same_type)
        self.lambda_H(attrs)
        self.higgs_potential_derivative(attrs)
        self.d_phi(attrs, neighbor_pm_val_same_type)
        self.phi(attrs)
        self.h(attrs)
        self.energy_density(attrs)

    
    def d_phi(self, attrs, neighbor_pm_val_same_type):
        d_phi = self._dX(
            attrs=attrs,
            d=self.env["d"],
            neighbor_pm_val_same_type=neighbor_pm_val_same_type,
            field_key="h"
        )
        print(f"∂μh(x):{d_phi}")
        return d_phi

    
    def phi(self, h, vev):
        """
        h = attrs["h"]
        vev = attrs["vev"]
        """
        new_phi = (1/np.sqrt(2)) * np.array([0, vev + h])
        print(f"{self.symbol}: {new_phi}")
        return new_phi

    
    def _mass_term(self, h, mass):
        return -mass ** 2 * h

    
    def h(self, h, mass, h_prev, laplacian_h, dV_dh):
        # Klein-Gordon-Update (explizit)
        """
        h=attrs["h"]
        h_prev= attrs["h_prev"]
        laplacian_h=attrs["laplacian_h"]
        dV_dh=attrs["dV_dh"]
        d=self.env["d"]
        mass=attrs["mass"]
        """
        mass_term = self._mass_term(h, mass)
        h = 2 * h - h_prev + self.env["d"]["t"] ** 2 * (laplacian_h + mass_term - dV_dh)
        print(f"h(x): {h}")
        return h

    
    def higgs_potential_derivative(
            self,
            vev,
            lambda_h,
            h,
    ):
        """
        vev=attrs["vev"]
        lambda_h=attrs["lambda_h"]
        h=attrs["h"]
        """
        mu = vev * np.sqrt(lambda_h)
        dV_dh = -mu ** 2 * (vev + h) + lambda_h * (vev + h) ** 3
        print(f"∂V/∂h = {dV_dh}")
        return dV_dh

    
    def lambda_H(self, mass,vev):
        """
        Higgs-Selbstkopplung
        λh bestimmt, wie steil der Mexican Hat ist
        λ_H = m_H² / (2 * v²)
        """
        """mass=attrs["mass"]
        vev=attrs["vev"]"""
        lambda_h = (mass ** 2) / (2 * vev ** 2)
        print(f"λ_H = {lambda_h}")
        return lambda_h

    
    def energy_density(self, d_phi, mass,h,vev,laplacian_h):
        """
        Lokale Energiedichte des Feldpunktes:
        E = 0.5*(∂_t h)^2 + 0.5*(∇h)^2 + V(h)
        Hamiltonian = k*(L0^2/2 + 2*l1*(l1*cos(q1) - L0)*cos(q1)) - g*(l1*m1*sin(q1) + l2*m2*cos(q2)) + (m3*a*a + m1*l1*l1*p2*p2 + 4*b*(b*(m2 + m3) + m3*a*cos(q2)))/(2*l1^2*l2^2*m3*(m1 + 4*(m2 + m3*sin(q2)^2)*sin(q1)^2))
        """
        kinetic = 0.5 * (d_phi[0]) ** 2  # Energie, die durch zeitliche Änderungen entsteht
        gradient = 0.5 * sum([c ** 2 for c in d_phi[1:]])
        potential = self._higgs_potential(mass,h,vev,laplacian_h)

        s_gradient = np.abs(gradient)

        energy = np.array(kinetic + s_gradient + potential).tolist()
        # LOGGER.info("self.energy updated", self.energy, type(self.energy))
        print(f"E = {energy}")
        return energy

    
    def _higgs_potential(self, mass,h,vev,laplacian_h):
        """
        V(h) = 0.5 m^2 h^2 + λ v h^3 + 0.25 λ h^4
        """
        """
        m2 = attrs["mass"] ** 2
        v = attrs["vev"]
        l = attrs["lambda_h"]
        h = attrs["h"]
        """
        m2 = mass**2
        h_potential = (
            0.5 * m2 * h ** 2
            + laplacian_h * vev * h ** 3
            + 0.25 * laplacian_h * h ** 4
        )
        return h_potential

    
    def laplacian_h(
            self,
            h,
            neighbor_pm_val_same_type,
    ):
        laplacian_h = 0
        for key, item in neighbor_pm_val_same_type.items():
            item:[tuple, tuple]
            p_item = item[0]
            m_item = item[1]

            # extract value
            val_plus = p_item[1]
            val_minus = m_item[1]

            laplacian_h += (val_plus + val_minus - 2 * h) / self.env["d"][key] ** 2
        return laplacian_h

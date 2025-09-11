import numpy as np

from qf_core_base.fermion.ferm_base import FermionBase
from utils._np.serialize_complex import deserialize_complex


class FermionTime(FermionBase):

    """
    Experimental class to rebuild the SM with time as absolute force

    time = e + pos (3D)

    Es gibt zeitliche ableitung von mir selbst (unterscheidliche zeitwege)
    und unterschieldiche zeitwerte im raum aus welchem mandie räumliche zeitableitung berechenn kann
    """


    def __init__(self, d, attr_keys, theta_W, time, e, pos):
        super().__init__(d, attr_keys, theta_W)
        self.pos=pos
        self.e=e
        self.time=time


    def calc_time(self):
        return self.e + np.sum(self.pos)


    def d_time(
            self,
            field_forward,
            field_backward,
    ):
        """
        Zeitliche Ableitung (rebuild) in beliebige Richtung
        """
        return (field_forward - field_backward) / self.time

    def dmu_time(self, neighbor_pm_val_same_type=None) -> float or int:
        """
        berechent zeitliche ableitung in alle richungen summiert das ergebniss unt eilt es durch die anzahl der berücksichtigten zeitrichtungen

        :param attrs:
        :param d:
        :param neighbor_pm_val_same_type:
        :param field_key:
        :return:
        """
        total_derivation = []
        for i, (key, item) in enumerate(neighbor_pm_val_same_type.items()):  # x,y,z
            # item:[tuple, tuple]
            # unpack single p/m tuple
            p_item = item[0]
            m_item = item[1]

            # extract value
            val_plus = deserialize_complex(p_item[1])
            val_minus = deserialize_complex(m_item[1])

            dmu_t = self.d_time(val_plus, val_minus)
            total_derivation.append(dmu_t)

        return np.sum(np.array(total_derivation)) / len(total_derivation)





    def dirac_time(self):
        pass

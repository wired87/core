import numpy as np

from qf_core_base.qf_utils.field_utils import FieldUtils


class SymBase(FieldUtils):

    """
    Base class for equations to run with all symmetry groups
    """
    def __init__(self):
        super().__init__()

    def _dmu(self, field_forward, field_backward, d, time=False):
        if time is False:
            # convert neighbor phi to complex
            ##print("phi_forward, phi_backward", phi_forward, phi_backward)
            field_forward = np.array(field_forward, dtype=complex)
            field_backward = np.array(field_backward, dtype=complex)
        else:
            # phi already complex
            pass

        single_dphi = (field_forward - field_backward) / (2 * d)
        return single_dphi


    def _dmu(self, self_ntype, attrs, d, neighbors_pm, field_key="h"):
        #print("neighbors_pm", neighbors_pm)

        phi_t = self._dmu(
            attrs[field_key],
            attrs[f"prev_{field_key}"],
            d["t"],  # dt = timestep
            time=True
        )

        dmu = [
            phi_t
        ]

        ##print("neighbors", self.neighbors_pm)
        for i, (key, pm) in enumerate(neighbors_pm.items()):  # x,y,z
            #print("Dphi run", i)
            plus_id = pm[0]
            minus_id = pm[1]

            neighhbor_plus = self.g.get_single_neighbor_nx(plus_id, self_ntype.upper())
            neighhbor_minus = self.g.get_single_neighbor_nx(minus_id, self_ntype.upper())

            phi_plus = neighhbor_plus[1][field_key]
            #print(f"nplus {minus_id}-> {neighhbor_plus[1]}:{phi_plus}")

            phi_minus = neighhbor_minus[1][field_key]
            #print(f"nminus {plus_id}-> {neighhbor_minus[1]}:{phi_minus}")

            if phi_plus is None:
                phi_plus = attrs[field_key]
            if phi_minus is None:
                phi_minus = attrs[field_key]

            dmu_x = self._dmu(phi_plus, phi_minus, d[key])
            #print(f">>>dmu{i}:", dmu_x)
            dmu.append(dmu_x)

        #print(f"Finished dmu forkey: {field_key}")
        return dmu
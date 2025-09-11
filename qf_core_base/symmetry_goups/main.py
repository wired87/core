import numpy as np

from qf_core_base.qf_utils.field_utils import FieldUtils
from qf_core_base.symmetry_goups.su2 import SU2
from qf_core_base.symmetry_goups.su3 import SU3
from qf_core_base.symmetry_goups.u1 import U1

from utils.utils import Utils


class SymMain(FieldUtils):

    def __init__(self, groups):
        # Init Symmetry Group Class
        super().__init__()
        self.sym_classes = {}
        #print("groups", groups)
        for group in groups:
            if group.upper() == "U(1)_Y":
                self.sym_classes["U(1)_Y"] = U1()
            if group.upper() == "SU(2)_L":
                self.sym_classes["SU(2)_L"] = SU2()
            if group.upper() == "SU(3)_C":
                self.sym_classes["SU(3)_C"] = SU3()
            self.u = Utils()


    def _calc_coupling_term_G(self, sym_group, nattrs, total_g_coupling, attrs, is_quark, is_gluon):
        if is_quark:
            for i in range(3):
                if is_gluon is True:


                    # Call method
                    coupling_term = self.sym_classes[sym_group].coupling_term(
                        **attrs,
                        **nattrs,
                    )
                    ct_array = np.array(coupling_term)
                    total_g_coupling[i] += ct_array

        else:
            coupling_term = self.sym_classes[sym_group].coupling_term(
                **attrs,
                **nattrs,
            )

            ct_array = np.array(coupling_term)
            total_g_coupling += ct_array

        #print(f"# _yukawa_couping_process finished: {total_g_coupling}")
        return total_g_coupling






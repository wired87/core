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


    def _yukawa_couping_process(self, nattrs, yukawa_total_coupling, is_quark, y, attrs):
        #print("nattrs", nattrs)
        if is_quark:
            for i in range(3):
                yukawa_term = self.yukawa_term(
                    y,
                    self.u.getr(nattrs, "h"),
                    self.u.getr(attrs, "psi_bar",),
                    self.u.getr(attrs, "psi"),
                    is_quark
                )
                yukawa_total_coupling[i] += yukawa_term[i]

        else:
            yukawa_term = self.yukawa_term(
                y,
                self.u.getr(nattrs, "h"),
                self.u.getr(attrs, "psi_bar"),
                self.u.getr(attrs, "psi"),
                is_quark
            )
            yukawa_total_coupling = yukawa_term
        #print(f"# _yukawa_couping_process finished: {yukawa_total_coupling}")
        return yukawa_total_coupling


    def yukawa_term(self, y, h:float, psi_bar, psi, is_quark, **attrs):
        """
        Nach SSB ist ℎ ein Skalar
        Physikalischer Yukawa-Term nach SSB:
        L = -(m_f / v) * h * (ψ̄ ψ)
        """

        y = float(y)
        #print("y, h, psi_bar", y, h, psi_bar)
        #print("psi, is_quark", psi, is_quark)
        if is_quark:
            """
            Quark besteht aus 3 arrays (farbadungen) bei welchem jede einen dirac 
            spinor repräsentiert (leptons tragen nur einen dirac spinor)
            """
            # psi bar und üsi ahben jeweils 3 spinoren welche man verwendn muss
            result = [-y * h * np.vdot(psi_bar[i], psi[i]) for i in range(3)]
        else:
            # h kein skalar mehr
            result = [-y * h * np.vdot(psi_bar, psi)]
        return np.array(result, dtype=complex)
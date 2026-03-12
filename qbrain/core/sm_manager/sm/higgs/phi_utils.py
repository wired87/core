from typing import Union
import numpy as np
from qbrain.qf_utils.field_utils import FieldUtils

class HiggsUtils(FieldUtils):

    def __init__(self):
        FieldUtils.__init__(self)

    def init_phi(
            self,
            h: Union[float, np.ndarray],
            vev: float = 246.0
    ) -> list[list[complex]]:
        """
        Prepares the Higgs complex doublet as a Python list of lists.
        """
        # Create the two complex components: phi_1 (zero) and phi_2 (VEV + h)
        phi_list = [
            [0.0 + 0.0j],  # phi_1 component (complex zero)
            [(vev + h) + 0.0j]  # phi_2 component (real part holds VEV + h)
        ]

        # Apply the normalization factor (1/sqrt(2))
        factor = 1.0 / np.sqrt(2.0)

        # Apply factor to the list elements (manual list comprehension)
        normalized_phi_list = [
            [c * factor for c in row]
            for row in phi_list
        ]

        return normalized_phi_list

    def init_d_phi(self, s=True, a_ssb=True):
        """
        Initiiert dmu_phi nach SSB oder davor.
        Nach SSB = 4 Einträge (2 real, 2 imag).
        Vor SSB = 4 rein real.
        # dmu_phi hat exakt das selbe ormat wie der "h" parameter nach ssb,
        die werte für jede zahl werden für alle nachbarn auf ssummiert
        """
        #if a_ssb is True:
        # Nach SSB: 2 komplexe Komponenten (Beispielwerte)
        d_phi = np.zeros((4,2), dtype=complex)
        return d_phi

    ###############
    # COUPLINGS ###
    ###############


    def hg_coupling(self, field_value, g, phi):
        """
        Higgs -> Gauge anderer term/ Ergebniss als Gauge -> Higgs
        warum ein 4,2 shape
        jede Raumzeit-Komponente kompletten Dublett-Vektor
        """
        terms = np.zeros((4, 2), dtype=complex)
        for mu in range(4):
            terms[mu] = 1j * g * field_value[mu] * phi
        return terms
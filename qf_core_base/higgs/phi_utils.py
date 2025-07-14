
import numpy as np


from utils._np.serialize_complex import serialize_complex


class HiggsUtils:
    def __init__(self):
        pass

    def init_phi(self, h, serialize=True):
        # init phi after ssb = 4 reel = 2 re 2 im
        # vor ssb 4 re
        # higge immer complex wobei einer der wete immer vev trägt
        vev = 246.0

        phi = (1 / np.sqrt(2)) * np.array([
            [0.0 + 0.0j],
            [(vev + h) + 0.0j]
        ], dtype=complex)

        if serialize is True:
            ##print("psi before serialization", psi, psi.shape)
            phi = serialize_complex(com=phi)
        return phi

    def init_d_phi(self, serialize=True, a_ssb=True):
        """
        Initiiert dmu_phi nach SSB oder davor.
        Nach SSB = 4 Einträge (2 real, 2 imag).
        Vor SSB = 4 rein real.
        # dmu_phi hat exakt das selbe ormat wie der "h" parameter nach ssb,
        die werte für jede zahl werden für alle nachbarn auf ssummiert
        """
        if a_ssb is True:
            # Nach SSB: 2 komplexe Komponenten (Beispielwerte)
            d_phi = np.zeros((4,2), dtype=complex)
        else:
            # Vor SSB: 4 rein reale Komponenten
            d_phi = np.array([
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ], dtype=float)

        if serialize:
            d_phi = serialize_complex(com=d_phi)

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
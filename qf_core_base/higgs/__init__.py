"""
Das higgs feld muss nicht extern angeregt werden. es aht bereits überall den gelcuhen wert

Masse (m): Bestimmt seine Trägheit (Widerstand gegen Beschleunigung) und seine Ruheenergie (E=mc
2
 ).
Ladung (q): Z.B. elektrische Ladung, Farbladung (für Quarks).
Bestimmt, wie es an bestimmte Kräfte (z.B. Elektromagnetismus)
koppelt.
Spin (s): Ein intrinsischer Drehimpuls. Fermionen wie
Elektronen und Quarks haben typischerweise Spin 1/2.
Andere Quantenzahlen: Leptonenzahl, Baryonenzahl,
Isospin, etc., je nach Teilchen.

"""
import os

from utils.convert_path_any_os import convert_path_any_os
from utils.file._yaml import load_yaml

r"""
PHI_EQ = r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_sim\physics\quantum_fields\nodes\higgs\equations\higgs.yaml" if OS_NAME == "nt" else "qf_sim/physics/quantum_fields/nodes/higgs/equations/higgs.yaml"
PHI_PHI_EQ = r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_sim\physics\quantum_fields\nodes\higgs\equations\phi_phi_eq.yaml" if OS_NAME == "nt" else "qf_sim/physics/quantum_fields/nodes/higgs/equations/phi_phi_eq.yaml"
PHI_GAUGE_EQ = r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_sim\physics\quantum_fields\nodes\higgs\equations\phi_gauge_eq.yaml" if OS_NAME == "nt" else "qf_sim/physics/quantum_fields/nodes/higgs/equations/phi_gauge_eq.yaml"

PHI_EQ = load_yaml(PHI_EQ)
PHI_PHI_EQ = load_yaml(PHI_PHI_EQ)
PHI_GAUGE_EQ = load_yaml(PHI_GAUGE_EQ)
"""
HIGGS_PARAMS = r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_core_base\higgs\higgs_params.yaml" if os.name == "nt" else "qf_core_base/higgs/higgs_params.yaml"
HIGGS_PARAMS = load_yaml(HIGGS_PARAMS)






"""
Phi 
- name: phi[0]
  equation: "-1j * GP"
  returns: "phi[0]"
  description: "Returns first value of the higgs vector"
  unphysical: true
  type: list
  parameters:
    - name: phi
      type: float
      source: local

    - name: nphi
      type: np.array
      source: local


- name: phi[1]
  equation: "(vev + phi + 1j * G0) / sqrt(2)"
  type: list
  parameters:
    - name: phi
      type: float
      source: local

    - name: nphi
      type: np.array
      source: local



"""








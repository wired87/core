
from utils.convert_path_any_os import convert_path_any_os
from utils.file._yaml import load_yaml



r"""
FERM_HIGGS_EQ = r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_sim\physics\quantum_fields\nodes\fermion\equations\fermion_higgs_eq.yaml" if OS_NAME == "nt" else "qf_sim/physics/quantum_fields/nodes/fermion/equations/fermion_higgs_eq.yaml"
FERM_GAUGE_EQ = r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_sim\physics\quantum_fields\nodes\fermion\equations\fermion_gauge_eq.yaml" if OS_NAME == "nt" else "qf_sim/physics/quantum_fields/nodes/fermion/equations/fermion_gauge_eq.yaml"
PSI_PSI_EQ = r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_sim\physics\quantum_fields\nodes\fermion\equations\psi_psi_eq.yaml" if OS_NAME == "nt" else "qf_sim/physics/quantum_fields/nodes/fermion/equations/psi_psi_eq.yaml"
PSI_EQ = r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_sim\physics\quantum_fields\nodes\fermion\equations\psi_eq.yaml" if OS_NAME == "nt" else "qf_sim/physics/quantum_fields/nodes/fermion/equations/psi_eq.yaml"

FERM_HIGGS_EQ = load_yaml(FERM_HIGGS_EQ)
FERM_GAUGE_EQ = load_yaml(FERM_GAUGE_EQ)
PSI_PSI_EQ = load_yaml(PSI_PSI_EQ)
PSI_EQ = load_yaml(PSI_EQ)
"""

FERM_PARAMS = convert_path_any_os("qf_core_base/fermion/psi_fields.yaml")
PSI_UNIFORM = convert_path_any_os("qf_core_base/fermion/psi_uniform.yaml")

FERM_PARAMS = load_yaml(FERM_PARAMS)
PSI_UNIFORM = load_yaml(PSI_UNIFORM)
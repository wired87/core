
from utils.convert_path_any_os import convert_path_any_os
from utils.file._yaml import load_yaml

#GAUGE = r"C:\Users\wired\OneDrive\Desktop\Projects\Brainmaster\simulator\physics\quantum_fields\nodes\g\gauge.yaml" if OS_NAME == "nt" else "qf_sim/physics/quantum_fields/nodes/g/gauge.yaml"
#GAUGE_EQ = r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_sim\physics\quantum_fields\nodes\g\equations\gauge.yaml" if OS_NAME == "nt" else "qf_sim/physics/quantum_fields/nodes/g/equations/gauge.yaml"
#GAUGE_EQ = load_yaml(GAUGE_EQ)

GAUGE_FIELDS = convert_path_any_os("qf_core_base/g/gauge_fields.yaml")
GAUGE_FIELDS = load_yaml(GAUGE_FIELDS)







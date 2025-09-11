#
import os

from utils.file._yaml import load_yaml

QFLEXICON=r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_core_base\qf_utils\qf_lexicon.yaml" if os.name == "nt" else "qf_core_base/qf_utils/qf_lexicon.yaml"
QFLEXICON = load_yaml(QFLEXICON)

FIELD_METADATA=r"C:\Users\wired\OneDrive\Desktop\Projects\qfs\qf_core_base\qf_utils\field_metadata.yaml" if os.name == "nt" else "qf_core_base/qf_utils/field_metadata.yaml"
FIELD_METADATA = load_yaml(FIELD_METADATA)

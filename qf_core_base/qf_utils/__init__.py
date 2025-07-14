#
from utils.convert_path_any_os import convert_path_any_os
from utils.file._yaml import load_yaml

QFLEXICON=convert_path_any_os("qf_core_base/qf_utils/qf_lexicon.yaml")
QFLEXICON = load_yaml(QFLEXICON)

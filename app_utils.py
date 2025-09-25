import os

from bm.settings import TEST_USER_ID, TEST_ENV_ID
from utils.file._yaml import load_yaml

USER_ID = TEST_USER_ID
ENV_ID = TEST_ENV_ID

NCFG_PAYLOAD = {
        "blocks": [
            {
                **load_yaml(
                    r"C:\Users\wired\OneDrive\Desktop\Projects\qfs\qf_core_base\stim_cfgs\block_cfg_node.yaml" if os.name == "nt" else "qf_core_base/stim_cfgs/block_cfg_node.yaml"),
                "phase": [
                    load_yaml(
                        r"C:\Users\wired\OneDrive\Desktop\Projects\qfs\qf_core_base\stim_cfgs\phase_node_cfg.yaml" if os.name == "nt" else "qf_core_base/stim_cfgs/phase_node_cfg.yaml")
                ]
            }
        ]
    }
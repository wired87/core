import os

import requests

from app_utils import ENV_ID, FB_DB_ROOT

from fb_core.real_time_database import FBRTDBMgr
from utils.file._yaml import load_yaml


testing = False
trgt_vm_ws_port = 8000

if testing is True:
    req_type = "https"
    trgt_vm_ip = "cluster.clusterexpress.com"  # or your VM IP
    trgt_vm_endpoint = f"{ENV_ID}/root/"
else:
    req_type = "http"  # or "https"
    trgt_vm_ip = f"127.0.0.1:{trgt_vm_ws_port}"  # or your VM IP
    trgt_vm_endpoint = f"root/"  # Replace with your TEST_ENV_ID

trgt_vm_domain = f"{req_type}://{trgt_vm_ip}/{trgt_vm_endpoint}"

vars_dict = {
    "DOMAIN": os.environ.get("DOMAIN"),
    "USER_ID": os.environ.get("USER_ID"),
    "GCP_ID": os.environ.get("GCP_ID"),
    "ENV_ID": os.environ.get("ENV_ID"),
    "INSTANCE": os.environ.get("FIREBASE_RTDB"),
    "STIM_STRENGTH": os.environ.get("STIM_STRENGTH"),
}

auth_payload = {
    "type": "auth",
    "data": {
        "key": ENV_ID
    }
}

deploy_payload = {
    "type": "deploy",
    "data": {
        "stim_cfg": {}
    }
}

state_payload = {  # InboundPayload
    "data": {
        "type": "start",
    },
    "type": "state_change",
}


def ncfg_process():
    db_manager = FBRTDBMgr()
    db_root = FB_DB_ROOT

    nid = "ELECTRON_px_0"
    node_cfg_payload = {
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

    # Upsert ncfg
    db_manager.upsert_data(
        path=f"{db_root}/cfg/node/{nid}",
        data=node_cfg_payload,
    )

    # Update global state
    db_manager.upsert_data(
        path=f"{db_root}/global_states/",
        data={
            "min_node_cfg_created": True
        },
    )
    print("Handled node cfg successfully")


def activate(cfg=True):
    # AUTH PAYLOAD
    print(f"Requesting trgt: {trgt_vm_domain}->{auth_payload}")
    response = requests.post(trgt_vm_domain, json=auth_payload)
    print(f"Auth response: {response.json()}")

    ncfg_process()

    # STATE CHANGE
    print(f"Requesting trgt: {trgt_vm_domain}->{state_payload}")
    response = requests.post(trgt_vm_domain, json=state_payload)
    print(f"State Change response: {response}-{response.json()}")


if __name__ == "__main__":
    activate()

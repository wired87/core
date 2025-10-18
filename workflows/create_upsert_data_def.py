import os

import numpy as np

from fb_core.real_time_database import FBRTDBMgr
from utils.file._yaml import load_yaml


"""def create_process(user_id, world_cfg, env_id):
    print(locals())

    cluster_dim = world_cfg["cluster_dim"]

    g = GUtils(
        nx_only=True,
        g_from_path=None,
        user_id=user_id,
        enable_data_store=True
    )

    creator = CreateWorld(
        g,
        cluster_dim,
        env_id=env_id,
        world_type="bare",
        user_id=user_id,
        env_cfg=world_cfg,
    )

    #creator.main()
    upsert_env(
        user_id,
        env_id,
        world_cfg=world_cfg,
    )"""


def upsert_env(user_id, env_id, world_cfg):
    """
    Upsert created ENV AND FRONTEND RELATED to FB
    - LOGS
    - GLOB STATES
    - CFG
    """
    print("ENV ID", env_id)
    print("USER ID", user_id)
    database = f"users/{user_id}/env/{env_id}"
    datastore_dest = f"{database}/datastore/"
    logs_dest = f"{database}/logs/"
    metadata_dest = f"{database}/metadata/"
    g_state = f"{database}/global_states/"
    env_cfg_path = f"{database}/cfg/"
    print("DS dest", datastore_dest)
    instance = os.environ.get("FIREBASE_RTDB")

    fb_mgr = FBRTDBMgr()

    print("Upsert Metadata")
    upsert_metadata(world_cfg["cluster_dim"], metadata_dest)

    global_stuff = load_yaml(
        r"C:\Users\wired\OneDrive\Desktop\BestBrain\qf_core_base\qf_utils\global_states.yaml" if os.name == "nt" else "qf_core_base/qf_utils/global_states.yaml"
    )

    print("Upsert Global States")
    fb_mgr.upsert_data(
        path=g_state,
        data=global_stuff,  # todo add security
        list_entry=False
    )

    """print("Upsert Empty Datastor")
    db_manager.firebase.upsert_data(
        path=datastore_dest,
        data={},
        list_entry=False
    )"""

    print("Create Empty Logs Dir")
    fb_mgr.upsert_data(
        path=logs_dest,
        data={
            env_id: {"ready": False}
        },
        list_entry=False
    )

    print("Create Empty Cfg Dir")
    fb_mgr.upsert_data(
        path=env_cfg_path,
        data={
            env_id: {
                "world": world_cfg
            }
        },
        list_entry=False
    )

    # UPSERT ENV CFG -> ALREADY SAVED IN NODE
    print("ENV upserted")

def upsert_metadata(cluster_dim, metadata_dest, valid_ntypes):
    abs_iters = np.prod(cluster_dim)
    fb_mgr = FBRTDBMgr()

    for i in range():
        metadata_struct = {}
        nodes = [
            f"{subu}_px_{i}"
            for subu in valid_ntypes
        ]

        for nid in nodes:
            data = {
                "id": nid,
                "status": {
                    "state": "INACTIVE",
                    "info": "none"
                },
                "messages_sent": 0,
                "messages_received": 0
            }
            metadata_struct[nid] = data
        fb_mgr.upsert_data(
            path=metadata_dest,
            data=metadata_struct,
            list_entry=False,
        )
        print(f"Metadata iter upserted {i}/{abs_iters}")



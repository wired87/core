import os


from app_utils import USER_ID, ENV_ID
from fb_core.real_time_database import FirebaseRTDBManager
from qf_sim.dj.websockets.relay_station import EnvCreatorProcess
from utils.utils import Utils

from dotenv import load_dotenv
load_dotenv()



if __name__ == "__main__":
    """
    Get cfg
    run env creation process
    delete all pods
    """
    envs = []
    db_root = f"users/{USER_ID}/env/{ENV_ID}"
    db_manager = FirebaseRTDBManager(
        database_url=os.environ.get("FIREBASE_RTDB"),
        base_path=db_root,
    )

    cluster_root = ""

    env_creator = EnvCreatorProcess(
        USER_ID,
        Utils(),
        envs,
        db_manager,
        cluster_root,
        db_root
    )

    cfg_struct = {
        "world_cfg": env_creator.cfg_creator.env_cfg_default,
        "node_cfg": {
            "px_id": {
                "sid1": {
                    "max_value": 10,
                    "phase": []
                }
            }
        }
    }

    env_creator.main(cfg_struct=[cfg_struct])

import asyncio
import os

from app_utils import USER_ID, ENV_ID
from fb_core.real_time_database import FirebaseRTDBManager
from qf_core_base.qf_utils.all_subs import FERMIONS
from qf_sim.dj.websockets.relay_station import EnvCreatorProcess
from utils.graph.local_graph_utils import GUtils
from utils.utils import Utils

from dotenv import load_dotenv

load_dotenv()


def create_world_process(user_id=USER_ID, env_id=ENV_ID):
    print("DEBUG: Starting create_world_process()")
    envs = []
    db_root = f"users/{user_id}/env/{env_id}"
    print(f"DEBUG: Firebase RTDB base path: {db_root}")
    db_manager = FirebaseRTDBManager(
        database_url=os.environ.get("FIREBASE_RTDB"),
        base_path=db_root,
    )
    print("DEBUG: FirebaseRTDBManager initialized.")

    cluster_root = ""  # Assuming this is intentionally empty or set elsewhere
    print(f"DEBUG: Cluster root: '{cluster_root}'")

    env_creator = EnvCreatorProcess(
        USER_ID,
        Utils(),
        db_manager,
        cluster_root,
    )
    print("DEBUG: EnvCreatorProcess initialized.")

    world_cfg = env_creator.cfg_creator.env_cfg_default
    print(f"DEBUG: Default world config obtained: {world_cfg}")

    env_creator.world_cfg_process(
        world_cfg=[
            world_cfg
        ]
    )
    print("DEBUG: world_cfg_process finished.")
    print("Process finished")


def create_cfg():
    print("DEBUG: Starting create_cfg()")
    """
    
    """
    def build_G():
        print("DEBUG: Inside build_G()")
        user_id = os.environ.get("USER_ID")
        env_id = os.environ.get("ENV_ID")
        print(f"DEBUG: build_G - USER_ID: {user_id}, ENV_ID: {env_id}")

        g = GUtils(
            nx_only=True,
            g_from_path=None,
            user_id=user_id,
            enable_data_store=True
        )
        print("DEBUG: GUtils initialized.")

        db_manager = FirebaseRTDBManager(
            database_url=os.environ.get("FIREBASE_RTDB"),
            base_path=f"users/{user_id}/env/{env_id}",
        )
        print("DEBUG: db_manager for G initialized in build_G.")

        initial_data = db_manager._fetch_g_data()
        print(f"DEBUG: Fetched initial G data: ")

        # Build a G from init data and load in self.g
        g.build_G_from_data(initial_data, env_id)
        print("DEBUG: G built from initial data.")
        return g, db_manager

    def get_center_node(g):
        print("DEBUG: Inside get_center_node()")
        center_px: tuple = None
        for nid, attrs in g.G.nodes(data=True):
            ntype = attrs.get("type")
            if ntype == "PIXEL":
                center = attrs.get("center", False)
                if center is True:
                    center_px = (nid, attrs)
                    print(f"DEBUG: Found center pixel: {center_px}")
                    break  # Assuming only one center pixel
        if center_px is None:
            print("DEBUG: No center pixel found.")
        return center_px

    def get_cfg_destination_paths(g):
        print("DEBUG: Inside get_cfg_destination_paths()")
        all_fermids = []
        for nid, attrs in g.G.nodes(data=True):
            ntype = attrs.get("type")
            if ntype in FERMIONS:
                all_fermids.append(nid)
                print(f"DEBUG: Added fermion ID: {nid}")
        print(f"DEBUG: All fermion IDs")
        return all_fermids

    # Build G from ALL_SUBS
    g, db_manager = build_G()
    print("DEBUG: G and db_manager returned from build_G().")

    # todo get center node
    # MVP stimulate all nodees similar
    center_px: tuple = get_center_node(g)
    print(f"DEBUG: Center pixel after get_center_node(): {center_px}")
    if center_px is None:
        print("ERROR: Center pixel is None, cannot proceed with CFG creation.")
        return  # Exit if no center pixel
    print("Cfg upsertion finished")


default_cfg = {
        "max_value": 1,
        "phase": [
            {
                "max_val_multiplier": 5,
                "iterations": 20
            }
        ]
    }


def upsert_cfg(env_id, user_id, struct):
    # create cfg
    db_manager = FirebaseRTDBManager(
        database_url=os.environ.get("FIREBASE_RTDB"),
        base_path=f"users/{user_id}/env/{env_id}",
    )
    # upsert cfg
    print("DEBUG: Starting CFG upsertion.")
    for k, v in struct.items():
        upsert_path = f"users/{v['user_id']}/env/{v['env_id']}/cfg/{k}/"
        print(f"DEBUG: Upserting data for key {k} to path: {upsert_path} with data: {v}")
        db_manager.upsert_data(
            path=upsert_path,
            data=v,
        )
        print(f"DEBUG: Cfg upsertion finished for {k}")



if __name__ == "__main__":
    """
    Get cfg
    run env creation process
    delete all pods
    """
    # todo check for env_ids
    # todo include logic for batch sim ncfg
    print("DEBUG: Script started from main block.")
    try:
        create_world_process()
    except Exception as e:
        print(f"ERROR: Error creating World: {e}")

    try:
        create_cfg()
    except Exception as e:
        print(f"ERROR: Error creating NCG: {e}")
    print("DEBUG: Script execution finished.")



"""
Das was jetzt noch fehlt:
Node cfg prozess:
- 
docker lokal testing
kubernetes deployment


"""
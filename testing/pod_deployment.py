import asyncio
import os
import threading
import time

from app_utils import USER_ID, ENV_ID
from fb_core.real_time_database import FirebaseRTDBManager
from gke.build_admin import GKEAdmin
from qf_core_base.qf_utils.all_subs import FERMIONS
from qf_sim.world.create_env import EnvCreatorProcess
from utils.deserialize import deserialize
from utils.dj_websocket.handler import ConnectionManager
from utils.graph.local_graph_utils import GUtils

from dotenv import load_dotenv

from utils.utils import Utils

load_dotenv()

def create_world_process(world_cfg=None, user_id=USER_ID):
    print("DEBUG: Starting create_world_process()")
    env_creator = EnvCreatorProcess(
        user_id,
    )
    world_cfg = deserialize(world_cfg)

    print("DEBUG: EnvCreatorProcess initialized.")
    if world_cfg is None:
        world_cfg = env_creator.cfg_creator.env_cfg_default
    print(f"DEBUG: Default world config obtained: {world_cfg}")

    env_creator.create_world_process(
        world_cfg=[
            world_cfg
        ],
    )
    print("world_cfg_process finished.")
    return env_creator





class Connector:

    """
    After pod deployment :  Connect
    """

    def __init__(self, env_cfg, user_id):
        self.ready_sessions = []

        self.env_cfg = env_cfg
        self.user_id = user_id

        self.cluster_root = "cluster.botworld.cloud"

        self.instance = os.environ.get("FIREBASE_RTDB")

        self.utils = Utils()
        self.connection_manager = ConnectionManager()
        self.gke_admin = GKEAdmin()
        self.db_manager = FirebaseRTDBManager(
            database_url=self.instance,
        )

    async def connect_to_pods(self):
        """
        Monitor state till ready
        Connect to all pods
        save / return ips to connect to
        send auth payload
        """

        print("Establish connecgion to pods")

        all_pods = list(
                struct["deployment"]["metadata"]["name"]
                for struct in self.env_cfg.values()
            )

        # check globs ready
        await self.check_ready(self.env_cfg)

        self.start_connection_thread(
            pod_names=all_pods
        )
        print("All connections threads started")

    async def check_ready(self, env_ids:list[str]):
        print("Start ready Thread")

        def _connect():
            """
            Wait till all clusters are build up
            """
            ready_envs:list = []
            try:
                while len(ready_envs) < len(env_ids):
                    for env_id in env_ids:
                        print("_connect", env_id)
                        data = self.db_manager.get_data(
                            path=f"global_states/",
                            ref_root=f"users/{self.user_id}/env/{env_id}/",
                        )

                        if "global_states" in data:
                            ready: bool = data["global_states"]["ready"]
                            if ready is True:
                                self.ready_sessions.append(env_id)
                                ready_envs.append(env_id)
                        time.sleep(2)
                        print(f"{len(ready_envs)}/{len(env_ids)}")
                print("Finished Ready check")
                return True
            except Exception as e:
                print(f"Error chck for global state: {e}")
            return False
        #
        self.ready_thread = threading.Thread(
            target=_connect,
            name="GLOBAL_READY_THREAD",
            daemon=True
        )

        # Start Thread
        self.ready_thread.start()
        print("Threadstarted succesfuully")


    async def connect_all_pods_process(
            self,
            pod_names: list[str]
    ) -> list:
        print("Connection request process started")
        all_authenticated = []
        index = 0
        try:
            while len(all_authenticated) < len(pod_names):
                if index < 30:
                    for pod_name in pod_names:
                        success:bool = await self.connect_to_pod(
                            pod_name
                        )
                        if success is True:
                            all_authenticated.append(
                                all_authenticated
                            )
                        # Small delay between iters
                        time.sleep(1)
                        index += 1
                        print(f"{len(all_authenticated)}/{len(pod_names)} pods connected")
                else:
                    print("Max request attampts reached. Break process")
                    # Create List of missing pods that couldnt be connected to
                    missing_pods = [name for name in pod_names if name not in all_authenticated]
                    return missing_pods

            # return empty list if while loop finished
            return []

        except Exception as e:
            print(f"Error: {e}")
        print("Finished Connection request process")




    def start_connection_thread(self, pod_names):
        # FB Upsert thread
        print("Create Con thread")

        def _connect():
            missing_pods:list = asyncio.run(
                self.connect_all_pods_process(pod_names)
            )
            if len(missing_pods):
                # todo error intervention
                pass
            else:
                pass

        self.con_thread = threading.Thread(
            target=_connect,
            name="POD_INIT_CONNECTION",
            daemon=True  # Optional: Der Thread wird beendet, wenn das Hauptprogramm endet
        )

        # Start Thread
        self.con_thread.start()
        print("Connect to Pods thread started")


    async def connect_to_pod(self, pod_name):
        """
        Connect to a GKE cluster based on its ip:port
        :param ip:
        :param pod_name:
        :return:
        """

        auth_payload = {
            "type": "auth",
            "data": {
                "key": pod_name
            }
        }

        try:
            endpoint = f"{self.cluster_root}/{pod_name}/"
            cr = await self.utils.apost(
                url=endpoint,
                data=auth_payload,
            )
            if cr and "response_key" in cr and "key" in cr and "session_id" in cr:
                if cr["key"] == pod_name:
                    # Successful pod authenticated -> append valid
                    print(f"Pod {pod_name} connected successfully")
                    return True
                else:
                    print(f"Invlalid key received: {cr['key']}")
            else:
                raise ValueError("No con request triger controlled Exceptio")
        except Exception as e:
            print(f"Error fetching: {e}")
        return False











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

    """    
    try:
        create_cfg()
    except Exception as e:
        print(f"ERROR: Error creating NCG: {e}")
    """
    print("DEBUG: Script execution finished.")



"""
Das was jetzt noch fehlt:
Node cfg prozess:
- 
docker lokal testing
kubernetes deployment


"""
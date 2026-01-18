import asyncio
import os
import pprint
import json
from tempfile import TemporaryDirectory
from typing import List

from bm.settings import TEST_USER_ID

from type import NodeCFGType
from utils.deserialize import deserialize
from utils.id_gen import generate_id
from utils.utils import Utils

from dotenv import load_dotenv
load_dotenv()


# todo include finished head -> relay:
#  get ds
# todo listener global_states.finieshed

class EnvCreatorProcess:
    """
    Batch handles all received cfg files and creates envs from it

    WF:
    - create cluster - & env cfg files
    - Create ENVs from created ids
    - Deploy len(cfg) Images to GKE cluster (-> give env cfg)
    - Return session ids
    """

    # todo env size & cfg implement in request
    # spam init request
    def __init__(
            self,
            user_id,
    ):
        # todo split workld and node cfg process
        self.env_id = None
        self.db_root = None
        print("Initializing EnvCreatorProcess...")
        self.env_cfg = {}
        self.user_id = user_id
        self.resource_map = [
            "deployment",
            "service",
            "ingress",
        ]
        # CLASSES
        self.utils = Utils()

        self.file_store = TemporaryDirectory()


        # Set Creator URL
        self.request_type = "http" if os.name == "nt" else "https"
        self.domain = "127.0.0.1:8000" if os.name == "nt" else "bestbrain.tech"
        self.creator_subdomain = os.environ["BB_INFRA_SUB"]
        self.create_endp = f"{self.request_type}://{self.domain}/world/create/"
        print("EnvCreatorProcess initialized.")


    async def create_world_process(
            self,
            world_cfgs: List[dict] or None = None,
    ):
        world_cfgs:List[dict] = self.preprocess_world_cfg(
            world_cfgs
        )
        print(f"start world_cfgs")

        # CREATE ENV IDs
        world_cfgs:dict = self.create_env_ids(world_cfgs)

        # ADD ENV VARS TO CFG
        world_cfgs:dict = self.add_env_vars_to_cfg(world_cfgs)

        # CREATE GRAPH AND UPSERT DATA
        await self.world_creator(world_cfgs)
        # temrary creatioon process

        print("World creagtion finished")
        return world_cfgs


    def deploy_process(self, world_cfgs):
        print("===============DEPLOY CFGS==================")
        pprint.pp(world_cfgs)

        # DEPLOY GKE
        active_pods = self.gke_admin.deploy(
            deployment_struct=world_cfgs,
        )
        self.save_gke_cfg(world_cfgs)

        print("deploy_process finished")

        return active_pods


    def connect_process(self, env_ids):
        # CONNECT TO ALL PODS
        authenticated_pods = asyncio.run(
            self.gke_admin.connector.connect_all_pods_process(
                env_ids
            ))
        # Extend env globs struct
        self.update_env_glob_state(authenticated_pods)
        print("finished connect_process")


    def save_gke_cfg(
            self,
            cfg_struct, #app_name: worldcfg,...
    ):
        """
        SAVE CFG IN ENV CFG STRUCT
        """
        print("Save GKE cfg for depl & service")
        for app_name, struct in cfg_struct.items():
            try:
                db_root = f"users/{self.user_id}/env/{app_name}"
                # create enw space for env cfg
                path = f"{db_root}/cfg/world/"

                # UPSERT ALL CFG STRUCTS FOR ENV
                data = self.db_manager._check_keys(
                    struct
                )

                if "env" in data:
                    data.pop("env")

                self.db_manager.upsert_data(
                    path=path,
                    data=data
                )

                print(f"Resource CFG for {app_name} upserted")
            except Exception as e:
                print(f"Err save_gke_cfg: {e}")
        print("Save resources cfg process finished")



    def update_env_glob_state(self, authenticated_pods):
        if authenticated_pods:
            for env_id in authenticated_pods:
                self.db_manager.upsert_data(
                    path=f"users/{self.user_id}/env/{env_id.replace('-', '_')}/global_states/",
                    data={
                        "authenticated": True,
                        "cfgs_created": True,
                    }
                )
                print(f"Global state for env {env_id} updated")


    async def world_creator(self, world_cfgs:dict , local=True):
        # todo need to outsource creation logic to new pod
        data = self.create_creation_payload(world_cfgs)
        # mark: need this admin_data pckg for postrequest
        if local is True:
            for cfg in data:
                """create_process(
                    self.user_id,
                    cfg["world_cfg"],
                    cfg["env_id"]
                )"""
                pass
        else:
            await self.create_env_clusters(data)

    def node_cfg_process(self, node_cfg, env_id):
        print("handle node cfg")
        self.db_root = f"users/{self.user_id}/env/{env_id}"
        self.upsert_node_cfg(
            node_cfg,
        )
        print("Node cfg finished")



    def upsert_node_cfg(self, node_cfg: NodeCFGType):
        # todo drf view for async gathering -> o direkt in bq schon aufgesetzt
        print("Starting Firebase upsert for config admin_data...")

        for ferm_id, upsert_content in node_cfg["node_cfg"].items():
            # upsert_content = value, phase
            self.db_manager.upsert_data(
                path=f"{self.db_root}/cfg/{ferm_id}",
                data=upsert_content
            )
        print("Firebase upsert completed.")


    def create_env_ids(
            self,
            world_cfgs: list[dict],
    ) -> dict:
        """
        For each env list entry create an id
        """
        world_cfgs_struct = {}
        for world_cfg_item in world_cfgs:
            # Deserialize
            world_cfg_item = deserialize(world_cfg_item)

            # env id
            env_id = f"env_{self.user_id}_{generate_id(mixed_dt=False)}"
            world_cfgs_struct[env_id] = world_cfg_item
        return world_cfgs_struct

    def add_env_vars_to_cfg(
            self,
            world_cfgs: dict,
    ) -> dict:
        new_struct = {}
        print("Starting extension of config with environment and simulation variables...")
        for env_id, world_cfg_item in world_cfgs.items():

            # env vars
            env_variables:list[dict] = self.create_env_variables(
                env_id
            )

            new_struct[env_id] = {
                "env": env_variables,
                **world_cfg_item,
            }

            print(f"Config extended. New environment ID: {env_id}")
        print(f"ALL ENV CFGs created:")
        #pprint.pp(new_struct)
        return new_struct


    def create_env_variables(self, env_id, cfg:dict=None) -> dict:
        print("create_env_variables...")
        cfg = cfg or {}
        env_vars_dict = {
            "DOMAIN": "www.bestbrain.tech",
            "GCP_ID": "aixr-401704",
            "DATASET_ID": "QBRAIN",
            "BQ_DATA_TABLE": f"{env_id}_data",
            "LOGGING_DIR": "tmp/ray",
            "ENV_ID": env_id,
            "USER_ID": self.user_id,
            #"FIREBASE_RTDB": os.environ.get("FIREBASE_RTDB"),
            #"FB_DB_ROOT": f"users/{self.user_id}/env/{env_id}",
            "DELETE_POD_ENDPOINT": "gke/delete-pod/",
            #"GKE_SIM_CLUSTER_NAME": os.environ.get("GKE_SIM_CLUSTER_NAME"),
            "SG_DB_ID": env_id,
            #"GEMINI_API_KEY": os.environ["GEMINI_API_KEY"],
            **{
                k: v #(json.dumps(v) if not isinstance(v, str) else v)
                for k, v in cfg.items()
            },
        }
        pprint.pp(env_vars_dict)
        print("create_env_variables... done")
        return env_vars_dict



    async def create_env_clusters(self, data:list[dict]):
        """
        CREATE GRAPH AND UPSERT DATA
        :return:
        """
        print("Starting asynchronous creation of environment clusters...")
        try:
            response:list[dict] = await self.utils.apost_gather(
                self.create_endp,
                data
            )
            print("Created environment clusters asynchronously...")

            return response
        except Exception as e:
            print(f"Error creating env cluster: {e}")
        finally:
            print("Environment cluster creation process completed.")



    def create_creation_payload(
            self,
            world_cfgs:dict
    ):
        print("create_creation_payload")
        data = []
        for env_id, cfg in world_cfgs.items():
            data.append(
                {
                    "user_id": self.user_id,
                    "env_id": env_id,
                    "world_cfg": cfg,
                }
            )
        print("create_creation_payload finished")
        return data





if __name__ == "__main__":
    ec = EnvCreatorProcess(
        user_id=TEST_USER_ID,
    )

if __name__ =="__main__":
    world_cfgs = asyncio.run(ec.create_world_process())
    pod_names = ec.deploy_process(world_cfgs)
    env_ids = list(world_cfgs.keys())
    ec.connect_process(env_ids)

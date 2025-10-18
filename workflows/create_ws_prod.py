"""
Websocket logic around the creation process
"""
import asyncio
import json
import os
import pprint
import threading

from bm.settings import TEST_ENV_ID
from fb_core.real_time_database import FBRTDBMgr
from utils.dj_websocket.handler import ConnectionManager

class WorldCreationWf:

    def __init__(
            self,
            user_id,
            parent,
            cluster_root,
            g,
            instance,
            database,
            testing
    ):
        self.user_id=user_id
        self.parent=parent
        self.g=g
        self.connection_manager=ConnectionManager()
        self.cluster_root = cluster_root
        self.instance=instance
        self.database=database
        # Load demo G in g.G
        self.g.load_graph(
            local_g_path=self.g.demo_G_save_path
        )
        self.env_id_map = set()
        # Classe
        self.db_manager = FBRTDBMgr()
        self.db_manager.set_root_ref(self.database)
        self.testing = testing

    def node_cfg_process(self, data):
        node_cfg = data.get("node_cfg")
        env_id = data.get("env_id")
        # extend env_id
        env_cfg_path = f"users/{self.user_id}/env/{env_id}/cfg/"
        for ncfg in node_cfg:
            self.db_manager.upsert_data(
                path=env_cfg_path,
                data={
                    ncfg["id"]: ncfg
                },
            )
        print(f"NCFG for {env_id} set")




    async def world_cfg_process(self, data):
        """
        unpack raw cfg file
        start thead
        give it to env_creator
        """
        """await self.get_build_G_send_frontend(
            [TEST_ENV_ID])
        print("G Data sent")"""

        self.world_cfg_struct:list[dict] = data.get("world_cfg")

        print(f"world_cfg_process:")
        pprint.pp(self.world_cfg_struct)

        print("finit_handler")
        self.start_creation_thread()
        print("start_creation_thread")

        await self.parent.send(
            text_data=json.dumps(
                {
                    "type": "world_cfg",
                    "data": "Started creation successfully",
                }
            )
        )




    def start_creation_thread(self):
        """
        WF:
        Create
        connect
        save envs as dict
        """
        print("start creation thread")

        def run():
            asyncio.run(self.create_process(self.world_cfg_struct))

        self.data_thread = threading.Thread(
            target=run,
            #args=(world_cfg),
            name=f"CREATE_WORLD_THREAD-{self.user_id}",
            daemon=True,
        )
        self.data_thread.start()
        print("Thread started completely")

    async def create_process(self, world_cfg_struct:list[dict]):
        print("start create_process")
        content = {}

        for wcfg in world_cfg_struct:
            # extend env_id
            wcfg['id'] = f"env_{self.user_id}{wcfg['id']}"
            env_cfg_path = f"users/{self.user_id}/env/{wcfg['id']}/cfg/"

            self.db_manager.upsert_data(
                path=env_cfg_path,
                data={
                    "world": wcfg,
                    #"node": wcfg["node_cfg"]
                },
            )

            env_id = wcfg["id"]
            content[env_id] = wcfg["cluster_dim"]
            self.env_id_map.add(env_id)

        data = {
            "envs": content,
            "graph": await self.create_frontend_env_content(),
        }

        await self.parent.send(
            text_data=json.dumps({
                "type": "world_content",
                "data": data
            }
            ))


    async def build_graph(self):
        #if self.testing is True:
        # just send demo G for now to front for each
        if os.path.isfile(self.g.demo_G_save_path) is True:
            print("Build Demo G")
            self.g.load_graph(local_g_path=self.g.demo_G_save_path)
        """else:
            initial_data = self.db_manager._fetch_g_data(
                db_root=self.database
            )

            # Build a G from init data and load in self.g
            self.g.build_G_from_data(initial_data,)

        # EXTEND WITH METADATA
        self.metadata_process()"""


    def metadata_process(self):
        data = self.db_manager.get_data(
            path=f"{self.database}/metadata/"
        )

        all_sub_nodes: list[str] = self.g.get_nodes(
            filter_key="type",
            filter_value=[],
            just_id=True,
        )

        print("metadata received:", data.keys())
        if data:
            # print("data[metadata]", data["metadata"])
            for node_id, metadata in data["metadata"].items():
                meta_of_interst = {k: v for k, v in metadata.items() if k not in ["id"]}
                print(f"add {meta_of_interst} to {node_id}")

                # NODE FROM ALL_SUBS?
                node_valid:bool = self.validate_meta_nid(
                    node_id,
                    all_sub_nodes,
                )
                if node_valid is True or (self.testing and node_id == TEST_ENV_ID):
                    continue
                    # todo include extra helper node (logs)
                self.g.G.nodes[node_id]["meta"] = meta_of_interst

    def validate_meta_nid(self, node_id, all_sub_nodes):
        """
        Extend the hlper ndoe list with all helper
        nodes to exclude in the graph build up
        (todo helper nodes get accessible through the worker node)
        """
        if node_id in all_sub_nodes:
            return True
        return False




    async def send_creds_frontend(self, listener_paths):
        await self.parent.send(text_data=json.dumps({
            "type": "creds",
            "message": "success",
            "data": {
                "creds": self.db_manager.fb_auth_payload,
                "db_path": os.environ.get("FIREBASE_RTDB"),
                "listener_paths": listener_paths
            },
        }))



    async def create_frontend_env_content(self):
        nodes = {}
        edges = {}
        id_map = set()

        for nid, attrs in self.g.G.nodes(data=True):
            if attrs.get("type").lower() not in ["users", "user", "env"]:
                nodes[nid] = {
                    "id": nid,
                    "pos": attrs.get("pos"),
                    "meta": attrs.get("meta"),
                    "color": "#004999",
                    "logs": {}
                }
                id_map.add(nid)
        print("Nodes extracted", len(nodes.keys()))

        for src, trgt, attrs in self.g.G.edges(data=True):
            if (attrs.get("src_layer").lower() not in ["env", "user", "users"]
                    and attrs.get("trgt_layer").lower() not in ["env", "user", "users"]):
                edges[attrs["id"]] = {
                    "src": src,
                    "trgt": trgt,
                }

        print("Edges extracted", len(edges.keys()))

        env_content = {
            "edges": edges,
            "nodes": nodes,
        }
        return env_content
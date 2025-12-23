import os

import ray
import requests

from core._ray_core.utils.ray_validator import RayValidator
from core.app_utils import DOMAIN, FB_DB_ROOT, TESTING
from fb_core.real_time_database import FBRTDBMgr
from utils.file.main import GraphBuilder


@ray.remote
class FBRTDBAdminWorker:

    """
    Für datenbank sctions. lässt den main loop ungehindert
    aufgaben verarbeiten.

    sammelt daten und upserted.
    Hat nur direkte verbindugn zu state upsertergo sp o fb
    Für batch uploads
    """

    def __init__(self):
        self.host = {}
        self.bq_ds_id = os.environ.get("DATASET_ID")
        self.node_type = "FBRTDB_ADMIN_WORKER"
        self.state = "inactive"
        self.domain = "http://127.0.0.1:8000" if os.name == "nt" else f"https://{DOMAIN}"

        self.send_type = "http" if os.name == "nt" else "https"
        self.testing = TESTING
        self.batch_size = 1

        # Gehaltsabrechnung
        self.database = FB_DB_ROOT
        print(f"DB root: {self.database}")

        # g not global -> qfns need to push new items here
        self.db_manager = FBRTDBMgr()

        self.state = "active"
        print(f"DBWorker initialisiert")




    def set_ready(self, ready):
        print(f"Upsert READY: {ready}")
        self.db_manager.upsert_data(
            path=f"{self.database}/global_states/",
            data={"ready": ready},
            list_entry=False,
        )

    def ping(self):
        return True


    def get_world_cfg(self):
        session_id = os.environ.get("ENV_ID")
        env_cfg = None

        path = f'{self.database}/cfg/{session_id}/world/'
        print(f"Requesting World Cfg from path: {path}")

        if session_id:
            try:
                env_cfg = self.db_manager.get_data(
                    path=path,
                )

                if "sim_time" not in env_cfg:
                    # no child received filter cfg
                    env_cfg = list(env_cfg.values())[0]

                # todo rm after testng without env-vars in cfg
                if "env" in env_cfg:
                    env_cfg.pop("env")

                if env_cfg is None:
                    env_cfg = self.db_manager.get_data(
                        path=path,
                    )

            except Exception as e:
                print(f"Err get_world_cfg: {e}")
        print("World cfg received:", env_cfg)
        return env_cfg


    async def batch_upsert(self, payload):
        session_id = os.environ.get("SESSION_ID")
        # DB bleibt be 1 table pro feld typ. session_id == identifier für jede entry
        print("Received batch upsert request")
        try:
            data:list = payload["admin_data"]
            data_type: "nodes" or "edges" = payload["type"]

            if data_type == "nodes":
                example_item = data[0]
                table_id = example_item["type"]
                schema = list(example_item.keys())

                # add session_id
                for d in data:
                    d.update({"session_id": session_id})

            else:
                table_id = "edges"
                schema = [k for item in data for k in item.keys()]

            schema.append("session_id")

            # Send upsertion request
            receiver = f"{self.domain}/bq/upsert/"
            data=dict(
                schema=schema,
                table=table_id,
                dataset_id=self.bq_ds_id
            )
            print("Data set -> push...")
            response = requests.post(receiver, data)
            print(f"Upsert response: {response}")
        except Exception as e:
            print(f"Erro upsert batch: {e}")

    def call(self, method_name, **kwargs):
        """for k,v in kwargs.items():
            print(f"{k}:{v}")"""
        if hasattr(self.db_manager, method_name):
            return getattr(self.db_manager, method_name)( **kwargs)
        else:
            print(f"Method {method_name} dies not exists in g or qfu")


    def get_global_state(self) -> dict:
        global_states = self.db_manager.get_data(
            path=f"{self.database}/global_states/",
        )
        for k, v in global_states.items():
            print("unpacked globs:", v)
            return v

    def get_data(self, path, child=False):
        data = self.db_manager.get_data(
            path=path,
        )
        #print("admin_data received from fb:", admin_data)
        if data:
            if child is True:
                for k, v in data.items():
                    print("unpacked globs:", v)
                    return v
        return data

    def get_db_manager(self):
        return self.db_manager

    async def iter_upsert(
            self,
            attrs,
            path,
    ):
        try:
            any_path_exists = False
            if path is not None:
                any_path_exists = True
                #print(f"Upsert to {path}:")
                #pprint.pp(attrs)
                self.db_manager.upsert_data(
                    path=path,
                    data=attrs,
                    list_entry=False,
                )
            if any_path_exists is False:
                print("no upsert path -> cancel operation.")
        except Exception as e:
            print(f"Error upsert: {e}")

    async def meta_upsert(self, payload):
        try:
            path = payload["db_path"]
            meta = payload["meta"]
            print(f"Upsert metadata for worker: {meta.get('id')}")
            self.db_manager.upsert_data(
                path=path,
                data=meta
            )
        except Exception as e:
            print(f"Ex upserting metadata: {e}")


    async def build_G_local_or_fetch(self):
        """
        build G from admin_data set self ref and creates
        all sub nodes of a specific
        """
        # Build G and load in self.g todo: after each sim, convert the sub-ield graph back to this
        print("BUild Graph")
        self.ray_validator = RayValidator(
            host=self.host, g=None
        )
        self.graph_builder = GraphBuilder()
        env = await self.graph_builder.build_graph()
        print("DB WOrkere finished build_G_local_or_fetch")
        return env




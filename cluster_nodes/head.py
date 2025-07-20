# pip install "_ray_core[serve]"
import os

import ray
from ray import serve
import json
from fastapi import WebSocket

from cluster_nodes.cluster_utils.G import UtilsWorker
from cluster_nodes.cluster_utils.db_worker import DBWorker
from cluster_nodes.cluster_utils.listener import Listener
from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from cluster_nodes.server.stat_handler import ClusterCreator

from cluster_nodes.server.state_handle import StateHandler
from cluster_nodes.server.types import HOST_TYPE, WS_INBOUND, WS_OUTBOUND

from utils.dj_websocket.handler import ConnectionManager
from utils.id_gen import generate_id
from utils.logger import LOGGER





@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": .2}
)
@serve.ingress(app)
class HeadServer:
    """
    Tasks:
    com with relay & QFN
    distribute commands to the right nodes
    wf:
    everything starts
    qfn dockers initialize and msg the HeadDepl (self).
    self message front
    """

    def __init__(self):
        LOGGER.info("Initializing HeadDepl...")
        self.session_id = "unknown"
        self.node_type = os.environ.get("NODE_TYPE")  # HEAD || QFN

        self.env_id = ENV_ID
        self.user_id = USER_ID

        self.ref = serve.get_deployment_handle(self.env_id)

        self.host: HOST_TYPE = {
            "head": self.ref,
            "field_worker": self.ref,
        }

        self.states = {}

        self.extern_host = {}
        self.messages_sent = 0
        self.manager = ConnectionManager()
        self.receiver = None
        self.attrs = None
        self.external_vm = None
        self.ws_key = None

        # Listen to DB changes
        self.listener = None
        self.all_worker_active = False
        self.all_subs = None
        self.state_acttion = {
            "start": self.start_action,
            "stop": self.handle_shutdown,
        }
        # start worker update loop
        self.state_checker = StateHandler.remote(
            head_ref=self.host["head"]
        )

        self._init_process()
        print("HeadDeplDeployment initialisiert!")

        # todo state from db listener

    async def state_handler(self, paylaod):
        # state for shutdown or active
        state = paylaod["data"]
        nid = paylaod["id"]
        if state == "active":
            self.state_acttion[state](nid, state)

    async def handle_shutdown(self, nid, state):
        self.g.get_all_subs_list(just_id=True)
        all_subs = self.g.get_all_subs_list()
        if nid and nid in all_subs:
            self.states["stop"].add(nid)

        if len(self.states["stop"]) == all_subs:
            LOGGER.info("All workers stopped, sending shutdown message to front")
            self.stop_action(all_subs)

    def start_action(self, all_nodes, state):
        ray.get(self.host["db_manager"].firebase.upsert_data.remote(
            path=self.global_states_listener_path,
            data={
                "total_nodes": len(all_nodes),
                "active_nodes": 100
            }
        ))

    def stop_action(self, all_subs):
        for nid, attrs in all_subs:
            attrs["ref"].exit.remote()
            LOGGER.info(f"Stopped worker {nid}")

    async def handle_all_workers_active(self):
        """
        Whaen everything is init send msg to front
        :return:
        """
        self.manager.active_connections[self.env_id].send_json({
            "data": {
                "status": "active"
            },
            "type": "status"
        })

    def _init_hs_relay(self, msg):
        key = msg["key"]
        if key == self.env_id:
            self.session_id = msg["session_id"]
            self.ws_key = generate_id()
            self.ws_key = key

    async def set_ws_validation_key(self, key):
        self.ws_key = key

    def get_active_env_con(self):
        return self.manager.active_connections.get(self.env_id, None)

    async def get_ative_workers(self):
        return self.states["active"]

    async def send_ws(self, data: WS_OUTBOUND, ptype: str):
        payload: WS_OUTBOUND = {
            "key": self.ws_key,
            "type": ptype,
            "data": data
        }
        LOGGER.info("Send payload to relay")
        con = self.get_active_env_con()
        await con.send_json(payload)

    async def set_all_subs(self, all_subs):
        if not len(self.all_subs):
            self.all_subs = all_subs
            LOGGER.info("ALL_SUBS set for head")

    def _init_process(self):
        print("init all HeadDepl classes")
        self.database = f"users/{self.user_id}/env/{self.env_id}/"
        self.instance = os.environ.get("FIREBASE_RTDB")

        self.host["utils_worker"] = UtilsWorker.options(
            name="utils_worker",
        ).remote()

        self.host["db_worker"] = DBWorker.remote(
            instance=self.instance,  # set root of db
            database=self.database,  # spec user spec entry (like table)
            user_id=self.user_id,
            host=self.host,
            attrs=self.attrs
        )

        # BUOLD G and load in utils_worker.G
        self.env = ray.get(self.host["db_worker"].build_G.remote(
            testing=True
        ))

        self.global_states_listener_path = f"{self.database}/global_states/"

        ## INIT CLASSES AND REMOTES ##
        # MSG Receiver any changes
        self.receiver = ReceiverWorker.remote(
            self.node_type,
            self.host,
            self.attrs,
            self.user_id,
            g=self.g,
        )

        # Listens to live state changes to distribute
        self.global_states_listener = Listener.remote(
            paths_to_listen=[
                self.global_states_listener_path
            ],
            db_manager=ray.get(self.host["db_worker"].get_db_manager.remote()),
            host=self.host
        )

        self.sim_state_handler = ClusterCreator(
            self.env,
            self.database,
            self.host,
            self.external_vm,
        )
        # Create and Load Ray Actors in the G
        self.sim_state_handler.load_ray_remotes()

        # BUILD G
        self.set_stuff()

        print("All classes in Head")

    def set_stuff(self):
        # Get STRUCT OF ALL SUBS STATES CATGORIZED IN QFNS
        self.states: dict = self.g.get_qf_subs_state()






# pip install "_ray[serve]"
import os

from ray import serve
import json
from fastapi import WebSocket

from cluster_nodes.cluster_utils.db_worker import DBWorker
from cluster_nodes.cluster_utils.listener import Listener
from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from cluster_nodes.server.env_ray_node import EnvNode
from cluster_nodes.server.stat_handler import ClusterCreator

from cluster_nodes.server.state_handle import StateHandler
from cluster_nodes.server.types import HOST_TYPE, WS_INBOUND, WS_OUTBOUND
from containers.head import app, ENV_ID, USER_ID
from qf_core_base.qf_utils.qf_utils import QFUtils

from utils.dj_websocket.handler import ConnectionManager
from utils.graph.local_graph_utils import GUtils
from utils.id_gen import generate_id
from utils.logger import LOGGER


def _validate_msg(msg) -> dict[WS_INBOUND] and bool:
    LOGGER.info("WS validated!")
    return json.loads(msg), True

@app.websocket("/{env_id}")
async def handle_ws(websocket: WebSocket, env_id):
    await websocket.accept()
    head = serve.get_deployment_handle(env_id)
    if head is not None:
        while True:
            try:
                async for data in websocket.iter_json():
                    data, valid = _validate_msg(data)
                    if valid is True:
                        await head.receiver.receive.remote(data)
            except Exception as e:
                print(f"Error while listening to: {e}")
                break
    else:
        await websocket.close()

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
        self.ws_key = None
        self.node_type = os.environ.get("NODE_TYPE")  # HEAD || QFN

        self.env_id = ENV_ID
        self.user_id = USER_ID

        self.ref = serve.get_deployment_handle(self.env_id)

        self.host: HOST_TYPE = {
            "head": self.ref,
            "field_worker": self.ref,
        }

        self.states = {}
        self.attrs = None
        self.g = None
        self.external_vm = None

        self.extern_host = {}
        self.messages_sent = 0
        self.manager = ConnectionManager()
        self.receiver = None

        # Listen to DB changes
        self.listener = None

        self.active_workers = []
        self.all_worker_active = False

        self.all_subs = None

        # start worker update loop
        self.state_checker = StateHandler.remote(
            head_ref=self.host["head"]
        )

        self._init_process()

        self.g = GUtils(
            nx_only=False,
            G=None,
            g_from_path=None,
            user_id=self.user_id,
        )

        self.qf_utils = QFUtils(
            self.g,
        )

        print("HeadDeplDeployment initialisiert!")



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
        return self.active_workers

    async def send_ws(self, data:WS_OUTBOUND, ptype:str):
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

        self.host["db_worker"] = DBWorker.remote(
            instance=self.instance,  # set root of db
            database=self.database,  # spec user spec entry (like table)
            g=self.g,
            user_id=self.user_id,
            host=self.host,
            attrs=self.attrs
        )
        # BUOLD G

        self.host["db_worker"].build_G.remote()
        self.env = self.g.G.nodes[ENV_ID]

        ## INIT CLASSES AND REMOTES ##
        # MSG Receiver any changes
        self.receiver = ReceiverWorker.remote(
            self.node_type,
            self.host,
            self.attrs,
            self.user_id,
            g=self.g,
        )

        self.listener = Listener.remote(
            self.g,
            self.host["db_worker"].get_db_manager.remote(),
            self.host
        )


        self.sim_state_handler = ClusterCreator(
            self.g,
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
        self.states:dict=self.g.get_qf_subs_state()



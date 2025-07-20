# pip install "_ray_core[serve]"
import os

import ray
from ray import serve
import json
from fastapi import WebSocket, FastAPI

from cluster_nodes.cluster_utils.listener import Listener
from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from cluster_nodes.manager.trainer import LiveTrainer
from cluster_nodes.server.env_ray_node import EnvNode
from cluster_nodes.server.set_endpoint import set_endpoint
from cluster_nodes.server.state_handle import StateHandler
from cluster_nodes.server.types import HOST_TYPE

from gdb_manager.g_utils import DBManager

from utils.dj_websocket.handler import ConnectionManager
from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER

# dynamic imports
"""


"""

app = FastAPI()

USER_ID = os.environ.get("USER_ID")
ENV_ID = os.environ.get("ENV_ID")
QFN_ID = os.environ.get("QFN_ID")


# 2. Definiere deinen Dienst als Ray Serve Deployment
# remote dynamisch übr hardware hoppen lassen (ip check)
@serve.deployment(
    route_prefix=set_endpoint(ENV_ID),
    num_replicas=1
)
@serve.ingress(app)
class QFNServerWorker:
    """
    Acts as QFN - Head of a specific Docker

    hält db, u utils
    startet alle remotes

    wf
    start


    """

    def __init__(self):
        self.node_type = "QFN"

        self.env_id = ENV_ID
        self.user_id = USER_ID
        self.id = QFN_ID
        self.host: HOST_TYPE = {}

        self.ref = serve.get_deployment_handle(self.id)
        self.head_ref = ray.get_actor(self.env_id)

        self.attrs = None
        self.g = None
        self.host = None
        self.external_vm = None

        self.extern_host = {}
        self.messages_sent = 0
        self.manager = ConnectionManager()
        self.receiver = None

        # Listen to DB changes
        self.listener = None

        self._init_process()
        self.active_workers = []
        self.all_worker_active = False

        self.all_subs = None

        # start worker update loop
        self.state_checker = StateHandler.remote(self.ref)
        self.state_checker.check_state.remote()

        print("ServerWorker-Deployment initialisiert!")

        # MARK: receiver will handle distribution of graph data
        # AND name all nodes

    async def handle_all_workers_active(self):
        self.manager.active_connections[self.env_id].send_json({
            "data": {
                "status": "active"
            },
            "type": "status"
        })

    @app.websocket("/{env_id}/{qfn_id}")
    async def main(self, websocket: WebSocket, env_id, qfn_id):
        # host_name = await self.manager.connect(env_id, websocket)
        while True:
            try:
                data = await websocket.receive_json()
                # await websocket.send_text(f"Message text was: {data}")
                # print(f"Msg received from {host_name}")
                # todo initial validation
                msg = self._validate_msg(data)
                self.receiver.receive.remote(msg)
            except Exception as e:
                print(f"Error while listening to: {e}")
                break

    def get_active_env_con(self):
        return self.manager.active_connections.get(self.env_id, None)

    def _validate_msg(self, msg):
        # todo more
        return json.loads(msg)

    async def get_ative_workers(self):
        return self.active_workers

    async def handle_active_worker_states(self, type, nid):
        if type == "append":
            self.active_workers.append(nid)
        elif type == "remove":
            self.active_workers.remove(nid)
        else:
            LOGGER.info(f"unexpected type in handle_active_worker_states: {type} from {nid}")

    async def set_all_subs(self, all_subs):
        if not len(self.all_subs):
            self.all_subs = all_subs
            LOGGER.info("ALL_SUBS set for head")

    async def _init_process(self):

        # Get self.ref & id
        self.host = {
            "id": self.env_id,
            "ref": self.ref,
            # self.parent and getattr doent work wirh rrs
        }

        self.database = f"users/{self.user_id}/env/{self.env_id}/"
        self.instance = os.environ.get("FIREBASE_RTDB")

        ## INIT CLASSES AND REMOTES ##
        # MSG Receiver any changes
        self.receiver = ReceiverWorker(
            self.node_type,
            self.host,
            self.attrs,
            self.user_id,
            G=self.g.G,
            extra_payload=None
        )
        # listen on stimulator changes in DB
        self.db_manager = DBManager(
            table_name="NONE",
            upload_to="fb",
            instance=self.instance,  # set root of db
            database=self.database,  # spec user spec entry (like table)
            nx_only=False,
            G=None,
            g_from_path=None,
            user_id=self.user_id,
        )
        self.g = GUtils(
            nx_only=False,
            G=None,
            g_from_path=None,
            user_id=self.user_id,
        )


        self.listener = Listener.remote(
            self.g.G,
            self.env_id,
            self.instance,
            self.database,
            self.user_id,
            upload_to="fb",
            table_name=None,
        )

        self.host.update({ # todo trainer, processor, visualizer etc
            "db_manager": self.db_manager
        })

        self.env_initializer = EnvNode(
            self.env_id,
            self.user_id,
            self.host,
            external_vm=False,
            session_space=None,
            db_manager=self.db_manager,
            g=self.g,
            database=self.database,
        )

        self.trainer = LiveTrainer.remote(
            self.g.G
        )

        # Fetch ds content and build G
        await self.env_initializer._init_world()

        # Now the locl Graph is build from db data

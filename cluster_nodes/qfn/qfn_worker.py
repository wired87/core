import os

import ray
from ray import serve

from cluster_nodes.cluster_utils.db_worker import DBWorker
from cluster_nodes.cluster_utils.listener import Listener
from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from cluster_nodes.manager.trainer import LiveTrainer
from cluster_nodes.server.env_ray_node import EnvNode

from container.head import ENV_ID, USER_ID
from gdb_manager.g_utils import DBManager

from utils.graph.local_graph_utils import GUtils
########
# TODO LATER -> CONCENTRATE ON FIELD WORKER


USER_ID = os.environ.get("USER_ID")
ENV_ID = os.environ.get("ENV_ID")
QFN_ID = os.environ.get("QFN_ID")

@ray.remote
class QFNWorker:

    def __init__(self, attrs):
        self.node_type = "QFN"
        self.attrs=attrs
        self.env_id = ENV_ID
        self.user_id = USER_ID
        self.id = QFN_ID
        self.head_ref = serve.get_deployment_handle(ENV_ID)
        self.ref = ray.get_actor(self.id)
        self.all_subs=None
        self.host = {
            "head": self.head_ref,
            "qfn": self.ref,
            #"db_worker": None
        }
        self.g = GUtils(
            nx_only=False,
            G=None,
            g_from_path=None,
            user_id=self.user_id,
        )


    async def _init_process(self):
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
            self.host["db_worker"].db_manager,
            self.host
        )


        self.host["trainer"] = LiveTrainer.remote(
            self.g.G
        )
        self.host["processor"] = None  # todo

        self.env_initializer = EnvNode(
            self.env_id,
            self.user_id,
            self.host,
            external_vm=False,
            session_space=None,
            db_manager=self.host["db_worker"],
            g=self.g,
            database=self.database,
            qfn_id=self.id,
        )
        # Fetch ds content and build G
        await self.env_initializer._init_world()
        await self.env_initializer.build_env()
        # ALL WOEKRS INTITIALIZE FROM THIS POINT
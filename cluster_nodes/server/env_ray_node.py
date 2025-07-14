import os

import ray

from cluster_nodes.cluster_utils.listener import Listener
from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from gdb_manager.g_utils import DBManager
from cluster_nodes.server.stat_handler import SimStateHandler
from qf_core_base.qf_utils.all_subs import ALL_SUBS
from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER

ALL_ENV_STATES = [
    "inactive"
    "active"
    "running"
]


class EnvNode:
    """
    Acts as head node utils

    workflow
    relay triggers vm start through gutils
    __init__ of script starts EnvNode & builds sys
    EnvNode establishes WS with Relay

    receives msg from relay

    Fetches data and build entire G
    updates all fields
    self.gdb_manager=DBManager(
            table_name=table_name,
            upload_to=upload_to,
            instance=instance,  # set root of db
            database=database,  # spec user spec entry (like table)
            nx_only=False,
            G=G,
            g_from_path=None,
            user_id=user_id,
        )


    sys läuft vorerst nur unter firebase als db!!
    """

    def __init__(
            self,
            env_id,
            user_id,
            host,
            external_vm,
            session_space,
            db_manager,
            g
    ):
        # todo ther is just a single node type -> role is dynamically

        self.state = "inactive"
        LOGGER.info("init ENV")
        self.g=g
        # Firebase endpoint for session data
        self.session_space=session_space
        self.env = {}
        self.host = host
        self.env_id = env_id
        self.user_id = user_id

        self.db_manager=db_manager
        self.initial_frontend_data = {}

        self.external_vm: bool = external_vm


    async def _init_world(self):
        """
        Turn on agents
        """
        # Build G and load in self.g todo: after each sim, convert the sub-ield graph back to this
        # format to save storage -> just for testing
        LOGGER.info(f"ENV _init_world request received")

        initial_data = self.db_manager._fetch_g_data()
        if initial_data is None:
            self.state = {
                "msg": "unable_fetch_data",
                "src": self.db_manager.database,
            }

        # Build a G from init data and load in self.g
        self.g.build_G_from_data(initial_data, self.initial_frontend_data, self.env_id)
        self.all_subs = [(nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") in ALL_SUBS]

        # when start -> send time as puffer + now = start to all start simulatnously

        self.env = self.g.G.nodes[self.env_id]
        self.env["ref"] = ray.get_runtime_context().current_actor

        # Create ray remotes from G data
        # await self.build_env()

        # Listen to DB cha  nges


        self.initial_frontend_data = {}
        self.local = True  #
        self.all_subs = None


        self.state = "active"
        LOGGER.info(f"ENV worker {self.env['id']}is waiting in {self.state}-mode")









    async def build_env(self):
        # Sim State Handler
        # build _ray network, start _qfn_cluster_node etc
        self.sim_state_handler = SimStateHandler(
            # g, env, database, host, external_vm, session_space
            self.g,
            self.env,
            self.database,
            self.host,
            self.external_vm,
            self.session_space,
        )
        # Create and Load Ray Actors in the G
        self.sim_state_handler.load_ray_remotes()

        # Check each nodes initialization state
        #ready = self.sim_state_handler._handshake() ->
        # each actor now send status request by its
        # own @ the end of init

        LOGGER.info("finished env build process")


"""

    def distribute_neighbors(self):
        for nid, attrs in [(nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") == "QFN"]:
            all_sub_fields = self.qf_utils.get_all_node_sub_fields(nid)

            # Loop all fields
            for sid, sattrs in all_sub_fields:
                # Load attrs in class
                LOGGER.info(f"get neighbors for: {sid}")

                # all sub-fild neighbors
                neighbors = self.g.get_neighbor_list(sid, ALL_SUBS)

                LOGGER.info("Send neighbors to remote")
                target = self.g.G.nodes[sid]["ref"]
                target.receiver.receive.remote(data={"type": "neighbors", "data": neighbors})


"""

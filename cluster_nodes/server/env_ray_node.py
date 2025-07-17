
import ray

from cluster_nodes.server.stat_handler import ClusterCreator
from qf_core_base.qf_utils.all_subs import ALL_SUBS
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
            host,

            env_id,
            user_id,
            external_vm,
            session_space,
            db_manager,
            g,
            database,
            qfn_id=None,
    ):
        # todo ther is just a single node type -> role is dynamically

        self.state = "inactive"
        LOGGER.info("init ENV")
        self.g=g
        self.qfn_id=qfn_id
        # Firebase endpoint for session data
        self.session_space=session_space
        self.env = {}
        self.host = host
        self.env_id = env_id
        self.user_id = user_id
        self.database=database
        self.db_manager=db_manager
        self.external_vm: bool = external_vm


    async def _init_world(self):
        """
        build G from data set self ref and creates
        all sub nodes of a specific
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
        self.g.build_G_from_data(initial_data, self.env_id)
        self.all_subs = [(nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") in ALL_SUBS]

        # when start -> send time as puffer + now = start to all start simulatnously

        self.env = self.g.G.nodes[self.env_id]
        self.env["ref"] = ray.get_runtime_context().current_actor

        # Create ray remotes from G data
        # await self.build_env()

        # Listen to DB cha  nges
        # reset
        self.initial_frontend_data = {}
        self.local = True  #
        self.all_subs = None

        self.state = "active"
        LOGGER.info(f"ENV worker {self.env['id']}is waiting in {self.state}-mode")


    async def build_env(self):
        # Sim State Handler
        # build _ray network, start _qfn_cluster_node etc
        self.sim_state_handler = ClusterCreator(
            # g, env, database, host, external_vm, session_space
            self.qfn_id,
            self.g,
            self.env,
            self.database,
            self.host,
            self.external_vm,
            self.session_space,
        )
        # Create and Load Ray Actors in the G
        self.sim_state_handler.load_ray_remotes()
        LOGGER.info("finished env build process")

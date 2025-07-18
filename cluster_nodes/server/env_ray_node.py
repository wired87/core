
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
        self.external_vm: bool = external_vm


    async def build_env(self):
        # Sim State Handler
        # build _ray network, start _qfn_cluster_node etc

        LOGGER.info("finished env build process")

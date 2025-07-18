
import ray

from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from cluster_nodes.qfn import ENV_ID
from qf_core_base.qf_utils.all_subs import ALL_SUBS

from gdb_manager.g_utils import DBManager
from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER

ALL_ENV_STATES=[
    "inactive"
    "active"
    "running"
]

@ray.remote
class DBWorker:

    """
    Für datenbank sctions. lässt den main loop ungehindert
    aufgaben verarbeiten
    """

    def __init__(
            self,
            g,
            attrs,
            instance,
            database,
            user_id,
            host,
            session_space=None,
            self_item_up_path=None,
            upload_to="fb",
            table_name=None,

    ):
        self.state = "inactive"
        self.g:GUtils = g

        self.host=host

        # Firebase endpoint for session data
        self.session_space = session_space
        self.self_item_up_path=self_item_up_path
        # g not global -> qfns need to push new items here
        self.db_manager=DBManager(
                table_name=table_name,
                upload_to=upload_to,
                instance=instance,  # set root of db
                database=database,  # spec user spec entry (like table)
                g_from_path=None,
                user_id=user_id,
            )

        self.allowed_hosts = [nid for nid, v in self.g.G.nodes(data=True) if v["type"] in ALL_SUBS]

        self.attrs = attrs

        self.receiver=ReceiverWorker.remote(
            cases=[
                ("upsert", self._handle_upsert),
                ("upsert_meta", self.iter_upsert),
            ]
        )
        self.state = "active"
        print(f"DBWorker initialisiert")

    def get_db_manager(self):
        return self.db_manager

    async def iter_upsert(self, attrs):
        self.db_manager.firebase.upsert_data(
            path=self.self_item_up_path,
            data=attrs
        )

    async def meta_upsert(self, payload):
        path = payload["db_path"]
        meta = payload["meta"]
        self.db_manager.firebase.upsert_data(
            path=path,
            data=meta
        )



    async def _session_upsert(self, data):
        sub_type = data["sub_type"]
        payload = data["data"]
        db_path = f"{self.session_space}{sub_type}/"
        if sub_type == "state":
            node_id = payload[0]
            state = payload[1]
            db_path = f"{db_path}{node_id}/"
            self.db_manager.firebase.upsert_data(
                path=db_path,
                item=state,
            )


    def build_G(self):
        """
        build G from data set self ref and creates
        all sub nodes of a specific
        """
        # Build G and load in self.g todo: after each sim, convert the sub-ield graph back to this
        # format to save storage -> just for testing
        LOGGER.info(f"Build G")

        initial_data = self.db_manager._fetch_g_data()
        if not initial_data:
            LOGGER.error("No initial data found, cannot build environment.")
            return

        # Build a G from init data and load in self.g
        self.g.build_G_from_data(initial_data, ENV_ID)

        LOGGER.info(f"Graph successfully build")

    async def _handle_upsert(self):
        pass


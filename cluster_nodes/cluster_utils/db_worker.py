
import ray

from cluster_nodes.cluster_utils.listener import Listener
from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from qf_core_base.qf_utils.all_subs import ALL_SUBS

from gdb_manager.g_utils import DBManager


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
            G,
            env,
            relay_id,
            instance,
            database,
            user_id,
            session_space,
            parent_ref,
            self_item_up_path,
            upload_to="fb",
            table_name=None,

    ):
        self.state = "inactive"

        # Firebase endpoint for session data
        self.session_space = session_space
        self.self_item_up_path=self_item_up_path
        # g not global -> qfns need to push new items here
        self.db_manager=DBManager(
                table_name=table_name,
                upload_to=upload_to,
                instance=instance,  # set root of db
                database=database,  # spec user spec entry (like table)
                nx_only=False,
                G=G,
                g_from_path=None,
                user_id=user_id,
            )

        self.allowed_hosts = [nid for nid, v in G.nodes(data=True) if v["type"] in ALL_SUBS]

        self.relay_id=relay_id
        self.attrs = env

        self.listener_worker = Listener.remote(
            self.g,
            parent_id,
            db_manager,
            parent_ref
        )

        self.receiver=ReceiverWorker.remote(
            cases=[
                ("upsert", self._handle_upsert),
                ("upsert_meta", self.iter_upsert),
            ]
        )
        self.state = "active"
        print(f"DBWorker initialisiert")


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










    async def _handle_upsert(self):
        pass


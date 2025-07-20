import ray

from cluster_nodes.server.types import LISTENER_PAYLOAD
from qf_core_base.qf_utils.all_subs import ALL_SUBS
from utils.graph.local_graph_utils import GUtils


@ray.remote
class Listener:
    """
    Listen to DB changes
    autonomous sends them to a parent

    Workflows included
    single upsert data never to
    """
    def __init__(
            self,
            paths_to_listen:list,
            db_manager,
            host,
            listener_type:str =None,
    ):
        self.host = host
        self.db_paths=self._get_db_paths()
        self.run=False

        self.db_manager=db_manager

        self.listener_paths = paths_to_listen

        self.db_manager.firebase._run_firebase_listener(
            db_path=self.listener_paths,
            update_def=self.listener_action,
            listener_type=listener_type
        )

        print("main updator intialized")

    async def listener_action(self, payload:LISTENER_PAYLOAD):
        self.host["field_worker"].receiver.receive.remote(
            payload=payload
        )




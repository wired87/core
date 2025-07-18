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
            g,
            db_manager,
            host
    ):
        self.host = host
        self.db_paths=self._get_db_paths()
        self.g:GUtils = g
        self.run=False

        self.db_manager=db_manager

        self.listener_paths = self.db_manager.firebase.get_listener_endpoints(
            nodes=self.g.id_map
        )

        self.db_manager.firebase._run_firebase_listener(
            db_path=self.listener_paths,
            update_def=self.listener_action
        )

        print("main updator intialized")



    async def listener_action(self, payload:LISTENER_PAYLOAD):
        self.host["field_worker"].receiver.receive.remote(
            payload=payload
        )


    def _get_db_paths(self):
        # get paths tolisten from
        paths = []
        for nid, attrs in [(nid, attrs) for nid, attrs in self.G.noes(data=True) if attrs["type"] in ALL_SUBS]:
            paths.append(f"{self.g.firebase.db_base}/{attrs['type']}/{nid}")
        return paths

import ray

from gdb_manager.g_utils import DBManager
from qf_core_base.qf_utils.all_subs import ALL_SUBS


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
            parent_id,
            instance,
            database,
            user_id,
            upload_to="fb",
            table_name=None,

    ):
        self.parent_id=parent_id
        self.db_paths=self._get_db_paths()
        self.g = g
        self.db_manager=DBManager(
            table_name=table_name,
            upload_to=upload_to,
            instance=instance,  # set root of db
            database=database,  # spec user spec entry (like table)
            nx_only=False,
            G=self.g.G,
            g_from_path=None,
            user_id=user_id,
        )
        self.run=False

        self.db_manager.firebase._run_firebase_listener(
            db_path=self.db_paths,
            update_def=self.main
        )
       #print("main updator intialized")

    def main(self, data):
        # todo queue with parallelization
        # Return changes to parent
       #print("DB change listened")
        trgt_node_ref = self.g.G.nodes[self.parent_id]["ref"]
        trgt_node_ref.receiver.receiver.receive.remote(data)
       #print(f"Data pushed to {self.parent_id}")


    def _get_db_paths(self):
        # get paths tolisten from
        paths = []
        for nid, attrs in [(nid, attrs) for nid, attrs in self.G.noes(data=True) if attrs["type"] in ALL_SUBS]:
            paths.append(f"{self.g.firebase.db_base}/{attrs['type']}/{nid}")
        return paths

import os

import ray

from qf_sim._qfn_cluster_node.nodes.db_worker import DBWorker
from qf_sim._qfn_cluster_node.nodes.receiver import ReceiverWorker
from qf_sim._qfn_cluster_node.nodes.updator_worker import UpdatorWorker
from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER
from utils.queue_handler import QueueHandler





@ray.remote
class FieldWorkerNode:
    """

    Repräsentiert ein QFN -> evtl switch later back to each field
    (elektron, myon etc) == 1 FieldWorkerNode ODER
    behalte es so bei aber erstelle für jedes feld einen

    Jeder self beinhaltet kopie dr nachbars werte
    ändert sich ein self.wert wird dieser zu allen nachbarn abgestoßen

    Kommunikation:
    Nachbarn: Direkt
    ENV: listener (push changes immer zur DB -> env lauscht)

    Workflow:
    Receive signal
    Run single iteration
    Return
    """

    def __init__(
            self,
            G,
            attrs: dict,
            env,
            user_id,
            database,
            host,
            external_vm,
            session_space,
            admin
    ):
        self.state = "inactive"
        self.env_id = env.get("id")
        self.env = env
        self.host=host
        self.user_id = user_id
        self.database = database
        self.external_vm = external_vm

        self.instance = os.environ.get("FIREBASE_RTDB")

        # Firebase endpoint for session data
        self.session_space=session_space

        # Build GUtils with ENV created G
        self.g = GUtils(
            nx_only=False,
            G=G,
            g_from_path=None,
            user_id=user_id,
        )  # -> DataManager -> local


        # self attrs
        self.attrs = attrs  # todo Get attrs directly from spanner/fbrtdb -> habe logik jetzt in
        self.type = attrs.get("type")
        self.id = attrs.get("id")
        self.parent = attrs["parent"][0]

        self.run = False
        self.admin=admin
        self.queue = QueueHandler()
        self.listener = None  # listen to chnges & update

        self.neighbors:tuple or None = None  # {id:ref # jeder nachbar erhält komplette Kopie

        self.connector = None  # all neighbors over ws -> zu kompliziert alle haben zugriff auf GUtils und somit auf direkte nachbarn
        self.history_hndler = None  # Klasse zum validieren letzter schritte

        # DB instance
        self.db_worker = DBWorker.remote(
            table_name="NONE",
            upload_to="fb",
            instance=self.instance,  # set root of db
            database=self.database,  # spec user spec entry (like table)
            nx_only=False,
            G=None,
            g_from_path=None,
            user_id=self.user_id,
            session_space=self.session_space,
        )

        self.main_loop_handler = UpdatorWorker.remote(
            self.g,
            self.attrs,
            self.env,
            self.id,
            self.parent
        )

        self.receiver = ReceiverWorker.remote(
            admin=self.admin
        )

        self.state = "standby"
        LOGGER.info(f"worker {self.id}is waiting in {self.state}-mode",)

        # Send status message
        self._upsert_session_data(
            payload=(self.id, self.state),
            sub_type="state",
        )



    async def _lampart_clock_handling(self, payload):
        """
        No global time
        Each time a msg gets rcvd,
        curretn node updates to the
        time of the sender jsut if:
        sender_time_stamp > self.time
        Wenn der sender_time_stamp von A kleiner ist als self.time, bedeutet das, dass Node B bereits Ereignisse verarbeitet hat, die kausal nach dem Senden der Nachricht von A lagen oder die parallel dazu verliefen, aber B schon weiter ist. In diesem Fall ist es entscheidend, dass B seine eigene Zeit beibehält und nur inkrementiert

        then: self.time + 1 to ensure both
        systems have now the same time
        """
        n_time = payload["time"]
        self.attrs["time"] = max(self.attrs["time"], n_time) + 1  # + 1 for causality









"""

    async def main(self):
        # Called after _init_nodes_process
        while self.run is True:
            # Cant
            all_updates = await asyncio.gather(*[
                attrs["ref"].update.remote(_time=self.env["time"])
                for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") in ALL_SUBS
            ])
            LOGGER.info("simulation finished")

            LOGGER.info("transfer data back to env")
            self.g.G.nodes[self.env.get("id")].receiver.receive.remote(
                data={
                    "type": "qfn_update",
                    "data": all_updates
                }
            )

"""
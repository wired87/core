import asyncio
import os

import ray

from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from cluster_nodes.cluster_utils.db_worker import DBWorker
from cluster_nodes.server.types import HOST_TYPE
from cluster_nodes.updator.updator_worker import UpdatorWorker
from gdb_manager.g_utils import DBManager
from qf_core_base.calculator.calculator import Calculator
from qf_core_base.qf_utils.all_subs import ALL_SUBS
from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER
from utils.queue_handler import QueueHandler



"""if os.name == "nt":
    from qf_sim.clusters.manager.db_worker import DBWorker
    from qf_sim.clusters.updator.updator_worker import UpdatorWorker
    from qf_sim.clusters.cluster_utils.receiver import ReceiverWorker
else:
    classes = [
        ("manager/db_worker.py", "DBWorker"),
        ("updator/updator_worker.py", "UpdatorWorker"),
        ("cluster_utils/reciever.py", "ReceiverWorker")
    ]
    for c in classes:
        module_path = c[0]
        class_name = c[1]
        globals()[class_name] = get_py_module_content(
            class_name=class_name,
            py_module_path=module_path,
        )"""


ENV_ID = os.environ.get("ENV_ID")
INSTANCE = os.environ.get("FIREBASE_RTDB")
NODE_TYPE = os.environ.get("NODE_TYPE")

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
            admin,
            neighbors,
    ):

        """
        G,
        attrs: dict,
        env,
        user_id,
        database,
        host,
        external_vm,
        session_space,
        admin
        """

        self.state = "online"
        self.G = G
        self.attrs = attrs
        self.env = env
        self.user_id = user_id
        self.database = database
        self.host: HOST_TYPE = host # include now: head & qfn ref
        self.external_vm = external_vm
        self.session_space = session_space
        self.admin = admin
        self.instance = INSTANCE
        self.host["node_worker"] = ray.get_runtime_context().current_actor

        self.neighbors:dict=neighbors
        # Build GUtils with ENV created G
        self.g = GUtils(
            nx_only=False,
            G=None,
            g_from_path=None,
            user_id=self.user_id,
        )

        self.type = self.attrs.get("type")
        self.id = self.attrs.get("id")

        self.parent = self.attrs["parent"][0]

        self.queue = QueueHandler()

        self.run = False
        self.listener = None  # listen to chnges & update
        self.connector = None  # all neighbors over ws -> zu kompliziert alle haben zugriff auf GUtils und somit auf direkte nachbarn
        self.history_hndler = None  # Klasse zum validieren letzter schritte

        self.calculator = Calculator(self.g)

        # DB instance
        self.db_path = f"{self.database}/{self.id}"

        self.host["db_worker"] = DBWorker.remote(
            instance = self.instance,  # set root of db
            database = self.database,  # spec user spec entry (like table)
            g = self.g,
            user_id = self.user_id,
            host = self.host,
            attrs = self.attrs
        )

        self.main_loop_handler = UpdatorWorker.remote(
            self.g,
            self.attrs,
            self.env,
            self.id,
            self.parent,
            host=self.host
        )

        self.receiver = ReceiverWorker.remote(
            main_loop_handler=self.main_loop_handler
        )

        self.attrs["metadata"]["status"]["state"] = "active"
        LOGGER.info(f"worker {self.id} is waiting in {self.state}-mode",)

        # Send status message
        self._upsert_metadata()


    async def apply_neighbor_changes(self, nid, attrs):
        LOGGER.info(f"Update attrs for n: {nid}")
        self.neighbors[nid].update(attrs)

    async def _upsert_metadata(self):
        # Send
        await self.host["db_worker"].receiver.receive.remote(
            payload={
                "db_path": f"{self.database}/metadata",
                "meta": self.attrs["metadata"],
                "type": "upsert_meta"
            }
        )
        LOGGER.info("Upsert process finished")

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

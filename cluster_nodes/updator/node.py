import asyncio
import os

import ray

from cluster_nodes.cluster_utils.listener import Listener
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

class DBStateChangeHandler:
    """
    Case: User stops the Sim

    Wf stop:
    user send stop command
    db upsert
    self.global_states_listener triggert global_changes methode im receiver
    global_changes stoppt uperator
    upsert fieldworker state to db
    head server recieves and collects all states
    shut them down
    """
    def __init__(self, host, database):
        self.host = host
        # Listens to live state changes to distribute
        self.global_states_listener = Listener.remote(
            paths_to_listen=[
                f"{database}/global_states/"
            ],
            db_manager=ray.get(self.host["db_worker"].get_db_manager.remote()),
            host=self.host,
            listener_type="global_change",
        )





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


    todo: listener für state changes
    """

    def __init__(
            self,
            g,
            attrs: dict,
            env,
            user_id,
            database,
            host,
            external_vm,
            admin,
            neighbor_struct,  # get listener paths and implement changes directly
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
        self.node_type = attrs.get("type")
        self.state = "online"
        self.attrs = attrs
        self.env = env
        self.user_id = user_id
        self.database = database
        self.host: HOST_TYPE = host  # include now: head & qfn ref
        self.external_vm = external_vm
        self.admin = admin
        self.instance = INSTANCE
        self.host["field_worker"] = ray.get_runtime_context().current_actor
        self.g = g

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
        self.states_db_path = f"{self.database}/global_states/"

        self.host["db_worker"] = DBWorker.remote(
            instance=self.instance,  # set root of db
            database=self.database,  # spec user spec entry (like table)
            g=self.g,
            user_id=self.user_id,
            host=self.host,
            attrs=self.attrs
        )

        self.updator_name = f"{self.id}_updator"
        self.main_loop_handler = UpdatorWorker.options(name=self.updator_name).remote(
            self.g,
            self.attrs,
            self.env,
            self.id,
            self.parent,
            host=self.host,
            neighbor_struct=neighbor_struct
        )
        self.host["updator_worker"] = self.main_loop_handler

        self.state_change_handler = DBStateChangeHandler(
            self.host,
            database
        )

        self.receiver = ReceiverWorker.remote(
            self.node_type,
            self.host,
            self.attrs,
            self.user_id,
        )

        # handle state
        self.state = "active"
        self.state_upsert()

        LOGGER.info(f"worker {self.id} is waiting in {self.state}-mode", )


    def state_upsert(self, state=None):
        self.attrs["metadata"]["status"]["state"] = self.state
        ray.get(self.host["db_worker"].iter_upsert.remote(self.attrs))


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



    async def get_state(self):
        return self.state
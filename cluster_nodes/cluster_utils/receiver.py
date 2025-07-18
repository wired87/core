import ray

from cluster_nodes.cluster_utils.msg_handlers.data_msg_handler import DataMessageManager
from cluster_nodes.cluster_utils.msg_handlers.head_msg_handler import HeadMessageManager
from cluster_nodes.cluster_utils.msg_handlers.qfn_msg_handler import QFNMsgHandler
from cluster_nodes.cluster_utils.msg_handlers.worker_msg_handler import WorkerMessageManager

from utils.queue_handler import QueueHandler

@ray.remote
class ReceiverWorker:
    def __init__(
            self,
            node_type,
            host,
            attrs,
            user_id,
            g=None,
            database=None,
    ):
        self.queue_handler = QueueHandler()
        self.running = False
        self.host_node_type = node_type
        self.host = host
        self.attrs=attrs
        self.user_id=user_id
        self.g=g
        self.database=database

        if self.host_node_type == "head":
            self.msg_handler = HeadMessageManager(
                attrs, user_id, self.database, g, host
            )
            self.cases:list[tuple]=[
                ("status", self.msg_handler._state_req),
                ("worker_status", self.msg_handler._state_handler),
                ("start", self.msg_handler._start),
                ("stop", self.msg_handler._stop),
                ("stim", self.msg_handler._stim_handler),
                ("cluster_msg", self.msg_handler.send_message),
                ("init_handshake", self.msg_handler._init_handshake),

                # WS
            ]

        elif self.host_node_type == "data_processor":
            self.msg_handler = DataMessageManager(
                parent_ref=self.host["field_worker"],
            )

            self.cases: list[tuple] = [
                ("data_update", self.msg_handler.get_data_update),
            ]

        elif self.host_node_type == "trainer":
            pass


        elif self.host_node_type == "QFN":
            self.msg_handler = QFNMsgHandler(
                self.host,
                self.attrs,
                self.user_id,
                self.g

            )

        else:
            self.msg_handler = WorkerMessageManager(
                host,
                attrs,
                user_id,
                self.g,
            )
            self.cases = [
                ("neighbors", self.msg_handler._set_neighbors),
                ("db_changes", self.msg_handler._set_neighbors),
                ("n_change", self.msg_handler._handle_n_change),
                ("status", self.msg_handler._state_req),
                ("start", self.msg_handler._start),
                ("stop", self.msg_handler._stop),
            ]
        print("ReceiverWorker initialized.")


    async def _validate_request(self):
        # todo
        return True

    async def receive(self, data):
        """
        For sim updates (stop, pause etc)
        """
        data_type = data["type"]

        ok = await self._validate_request()

        if ok:
            for case, action in self.cases:
                if data_type == case:
                    await action(data)
        else:
           print("validation failed by type", data_type)




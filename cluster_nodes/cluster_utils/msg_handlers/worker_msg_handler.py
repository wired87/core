from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER


class WorkerMessageManager:

    """
    Entry for any messages to worker nodes
    """

    def __init__(self, host, attrs, user_id, g):
        self.attrs=attrs
        self.attrs_id= attrs.get("id")
        self.user_id=user_id
        self.host=host

        self.g:GUtils = g



    async def apply_neighbor_changes(self, payload):
        LOGGER.info("Filte neighbor changes")
        # extract path
        path = payload["path"]
        attrs = payload["data"]

        if path.endswith("/"):
            path=path[:-1]

        nid = path.aplit("/")[-1]

        await self.host["parent"].apply_neighbor_changes(
            nid, attrs
        )


    async def _set_neighbors(self, payload):
        self.neighbors = payload


    async def _handle_n_change(self, payload):
        # qfn, env changes
        update_nnid = payload["id"]
        for i, (nnid, nattrs) in enumerate(self.neighbors):
            if nnid == update_nnid:
                await self._lampart_clock_handling(payload)
                self.neighbors[i][1].update(payload)
                break




    async def _upsert_session_data(self, payload, sub_type):
        LOGGER.info(f"send child msg: {payload}")
        # Send
        await self.db_worker._session_upsert.remote(
            data={
                "type": "cluster_msg",
                "sub_type": sub_type,
                "host": {
                    "id": self.attrs["id"]
                },
                "data": payload
            })
        LOGGER.info("Upsert process finished")





    async def _stop(self, payload):
        # stop running update loop
        self.main_loop_handler.stop.remote()
        # Just intern requests allowed
        sender_id = payload["host"]["id"]
        if sender_id == self.env["id"]:
            self.run = False
            # upload last state
            self.db_manager.firebase.upsert_batch(
                data={
                    f"{self.type}/{self.id}": self.attrs
                }
            )
            # History todo: there is jsut history. -> Fetch initial just freshest entry
            self.db_manager.firebase.upsert_history_batch()

    async def _state_req(self, payload):
        if payload == self.host_id:
            # Frontend Updator status request
            return (self.id, self.state)
        else:
            LOGGER.info(f"Mismatch request:self host id: {payload}:{self.host_id}")



    async def _start(self, payload):
        LOGGER.info(f"Start command received. start sim")
        start_time = payload["start_time"]
        self.main_loop_handler.main.remote(
            start_time
        )




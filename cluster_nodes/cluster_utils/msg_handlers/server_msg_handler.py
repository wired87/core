from ray import serve

from cluster_nodes.server.env_ray_node import EnvNode
from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER


class ServerMessageManager:
    def __init__(self, host, attrs, user_id,  extra_payload):
        self.attrs=attrs
        self.attrs_id= attrs.get("id")
        self.user_id=user_id
        self.host=host
        self.extra_payload=extra_payload
        self.g = GUtils(
            nx_only=False,
            G=None,
            g_from_path=None,
            user_id=self.user_id,
        )

    async def _relay(self, payload):
       #print("request received:", payload)
        EnvNode.receiver.receive.remote(
            data=payload
        )


    async def _cluster_msg(self, payload):
        # todo validation
        # todo fine granting of msg type (html, state etc...)
        ws = await self.host["ref"].get_active_env_con.remote()
        response = await ws.send_json(payload)
        LOGGER.info("Message transferred to front")


    async def _init_process(self, payload):
        self.env_id = payload.get("env_id")

        # Get self.ref & id
        self.host = {
                "id": payload.get("env_id"),
                "ref": serve.get_deployment_handle("1")
            }

        self.env_node = EnvNode.remote(
            self.env_id,
            self.host
        )
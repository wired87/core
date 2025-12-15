import logging
import os

import ray
from fastapi import Body
from ray import serve
from app_utils import APP, FB_DB_ROOT
from utils.id_gen import generate_id


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": .4},
    max_ongoing_requests=10
)
@serve.ingress(APP)
class Guard:
    """



    Guard of a docker container.
    Handles creation and storage of new anmespaces
    """
    def __init__(
            self,
            host,
    ):
        self.id = "GUARD"
        self.logger = logging.getLogger("ray.serve")
        self.logger.info("Initializing HeadDeployment...")

        self.namespace = {
            "example_id": {
                "object_store": None,
                "namespace_ref": None
            }
        }

        self.host = host
        self.database = FB_DB_ROOT

        # upsert online state db
        ray.get(
            self.host["DB_WORKER"].iter_upsert(
                path=f"/ADMIN_META/{self.id}",
                attrs={
                    f"state": "ONLINE"
                }
            )
        )

    @APP.post("/")
    async def post(self, payload: dict = Body()):
        self.logger.info(f"Guard: post message registered:{payload}")
        try:
            response = await self.handle_extern_message(payload)
            return {"status": "success", "data": response}
        except Exception as e:
            self.logger.error(f"Error while processing: {e}")
            data = e
        return {"status": "error", "data": data}


    async def handle_extern_message(self, payload):
        """
        Entry for all incoming & validated ws (or local) messages
        """
        try:
            self.logger.info(f"MSG FROM EXTERM RECEIVED: {payload}")
            payload_type = payload["type"]
            data = payload.get("data")
            if payload_type == "auth":
                return self._init_hs_relay(data)

            ##############

            else:
                self.logger.info(f"invalid request {payload_type}")
                return {"response": "Handshake pong"}
        except Exception as e:
            self.logger.error(f"Error in SERVE: {e}")

    def _init_hs_relay(self, data):
        """
        :param msg: key, sesion_id
        """

        self.logger.info("Guard: Init request received")

        if "realy_id" in data and "key" in data:  # and "env_vars" in data
            # Handshake - Key to save
            self.extern_key = data["key"]
            realy_id = data["realy_id"]
            self.intern_key = f"h{generate_id(20)}w{generate_id(15)}x"
            if realy_id == os.environ["RELAY_ID"]:
                return dict(
                    response_key=self.intern_key,
                    received_key=self.extern_key,
                )
        else:
            self.logger.info("Could not authenticate request:")
            return {
                "message": "Invalid request"
            }

    """def deploy_single_namespace(self, data):
        # node_cfg : pixel_utils, node_utils : list[id] : attrs, env_content.
        node_cfg = data["node_utils"]
        env_content = node_cfg["env_content"]

        self.deploy_workers(
            env_content,
        )

    def deploy_workers(
            self,
            env_content,
    ):
        self.logger.info("Deploy Workers")
        namespace_name = env_content["SESSION_ID"]

        host = self.create_host_store(env_content)

        ### MAIN ###
        self.logger.info(f"Init {namespace_name} in ray ")
        ray.get_actor(name="namespace_creator").create_namespace_main.remote(
            host.copy(),
            namespace_name,
        )



    def create_host_store(self, env_content):
        host: HOST_TYPE = {}

        # Load env vars in
        host["store"] = ray.put(env_content)
        return host


"""
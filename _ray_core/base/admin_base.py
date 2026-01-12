import sys
import os
import socket
import time
from typing import Dict, Any

import ray
import requests
from ray import serve, get_actor
from ray.exceptions import RayActorError
from ray.util.state import list_actors
from ray.util.state.common import ActorState

from core.app_utils import TESTING, ENV_ID
from _ray_core.base.base import BaseActor
from utils.run_subprocess import exec_cmd

OS_NAME = os.name

class RayAdminBase(BaseActor):

    def __init__(self):
        super().__init__()
        self.include_dashboard = OS_NAME != "nt"
        self.local_mode = OS_NAME == "nt"
        self.ip = socket.gethostbyname(socket.gethostname())
        self.disable = "0" if OS_NAME == "nt" else "1"
        self.ray_exe = os.path.join(os.path.dirname(sys.executable), "ray")
        print("RayBase initialized")

    def init_ray_process(self, serve=False):
        print("================== INIT RAY PROCESS ==================")
        self.stop_ray()
        self.start_head()
        self.init_ray()
        if serve is True:
            self.init_serve()


    def print_actor_states(self):
        self.status()
        self.list_tasks()
        self.list_actors(print_actors=True)
        self.timeline()

    def init_ray(self, namespace_name=None):
        os.environ["RAY_LOGGING_CONFIG_ENCODING"] = "JSON"
        print("init ray")
        for _ in range(10):
            try:
                ray.init(
                    ignore_reinit_error=True,
                    local_mode=False,
                    include_dashboard=True,
                    logging_config=ray.LoggingConfig(
                        encoding="TEXT" if TESTING is True else "JSON",
                        log_level="INFO",
                    ),
                    _temp_dir=self.ray_assets_dir,
                )
                break
            except Exception as e:
                print("Retrying ray.init()", e)
                time.sleep(1)

        print(f"ray initialized {ray.is_initialized()}")

    def get_actor_count_by_class(self, class_name: str) -> int:
        """
        Returns the count of active actors belonging to a specific class name.
        """
        try:
            # Get the state of all actors in the cluster
            all_actors: Dict[str, Any] = ray.state.actors()

            count = 0
            for actor_id, actor_info in all_actors.items():
                # Ray stores the class name in the 'class_name' field
                if actor_info.get("ClassID") == class_name:
                    count += 1

            return count
        except Exception as e:
            print(f"Error accessing Ray state: {e}")
            return 0

    def shutdown_sys(self):
        """
        Kill workers
        Delete cloud resource
        """
        try:
            # SET FINISH STATE DB
            ray.get_actor(name="FBRTDB").iter_upsert.remote(
                path=f"{os.environ['FB_DB_ROOT']}/global_states/",
                attrs={
                    "ready": False,
                    "finished": True,
                }
            )

            # Kill all workers
            all_actors: list[ActorState] = list_actors(detail=True)
            for actor in all_actors:
                name = actor.name
                # admin actors
                if name in ["RELAY", "HEAD"]:
                    continue
                ray.kill(get_actor(name=name))
                print(f"Actor {name} killed")

            # Send pod deletion request
            domain = os.environ.get("DOMAIN")
            endpoint = os.environ.get("DELETE_RCS_ENDPOINT")
            cluster_name = os.environ.get("CLUSTER_NAME")

            # Reset ENV
            #self.reset_env_vars()

            del_url = f"https://{domain}/{endpoint}"

            data = {
                #"pod_names": [self.session_id],
                "cluster_name": cluster_name,
                "env_id": ENV_ID,
            }

            response = requests.post(del_url, data=data)
            print("Delete response:", response)
            ray.actor.exit_actor()

        except Exception as e:
            print(f"Err shutdown_sys: {e}")
        print("Finished response")



    def start_head(self):
        ray_port = 6379
        _try = 0
        max_tries = 10
        for i in range(max_tries):
            print(f"try {i} to start head")
            try:
                cmd = [self.ray_exe, "start", "--head", f"--port={ray_port}", f"--temp-dir={self.ray_assets_dir}"]
                result = exec_cmd(cmd)
                if result is not None:
                    print("Started Head")
                    return
            except Exception as e:
                print(f"error start head: {e}")
                self.stop_ray()
            time.sleep(5)
        print("Head couldn be started")

    def stop_ray(self):
        try:
            print(exec_cmd([self.ray_exe, "stop", "--force"]))
            self.stop()
            print("Stopped existing ray processes")
        except Exception as e:
            print(f"error stop: {e}")

    def memory(self):
        # ray memory --stats-only
        exec_cmd([self.ray_exe, "memory", "--stats-only"])

    def start_serve(
            self,
    ):
        for i in range(10):
            try:
                print(f"[Try {i + 1}] Starting serve.run()")
                self.init_serve()
                print("✅ serve.start() started successfully")
                break
            except RayActorError as e:
                print(f"⚠️ serve.start() failed, retrying...: {e}")
                time.sleep(2)
            except Exception as e:
                print("🔥 Unexpected error in serve.run():", e)
                time.sleep(2)

    def init_serve(self):
        http_port = 8001
        serve.start(
            http_options={
                "host": "0.0.0.0",
                "port": http_port
            },
            detached=True,
            disable_dashboard=os.name == "nt",
        )


    def stop(self):
        if ray.is_initialized():
            ray.shutdown()
        print("ray shutdown")

    def status(self):
        exec_cmd([self.ray_exe, "status"])

    def list_tasks(self):
        exec_cmd([self.ray_exe, "list", "tasks"])

    def timeline(self):
        ray.timeline(filename="timeline.json")

    def create_static_docker_env_vars(self):
        return {
            "DOMAIN": "bestbrain.tech",
            "DATASET_ID": "QCOMPS",
            "GCP_ID": os.environ.get("GCP_PROJECT_ID"),
            "FIREBASE_RTDB": os.environ.get("FIREBASE_RTDB"),
        }


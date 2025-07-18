import os
import socket
import subprocess
import time

import ray
from ray import serve
from ray.exceptions import RayActorError

from bm.settings import OS_NAME
from cluster_container.head.main import HeadServer


class RayAdminBase:

    def __init__(self, env_id):
        self.logs_dir = r"C:\Users\wired\OneDrive\Desktop\BestBrain\tmp\ray\session_*\logs" if OS_NAME == "nt" else "/tmp/_ray/session_*/logs"
        self.include_dashboard = OS_NAME != "nt"
        self.local_mode = OS_NAME == "nt"
        self.ip = socket.gethostbyname(socket.gethostname())
        self.ray_port = 6379
        self.http_port = 8001
        self.env_id = env_id
        print("RayBase initialized")

    def start(self):
        os.environ["RAY_DISABLE_DASHBOARD"] = "1" if OS_NAME == "nt" else "0"

        for _ in range(10):
            try:
                ray.init(
                    ignore_reinit_error=True,
                    local_mode=self.local_mode,
                    include_dashboard=self.include_dashboard,
                    address=f"{self.ip}:{self.ray_port}",
                )

                break
            except Exception as e:
                print("Retrying ray.init()", e)
                time.sleep(1)
        print("ray initialized")

    def start_head(self):
        subprocess.run(["ray", "start", "--head", f"--port={self.ray_port}", "--include-dashboard=false"], check=True)

    def stop_ray(self):
        subprocess.run(["ray", "stop", "--force"], check=True)

    def start_serve(self):
        for i in range(10):
            try:
                print(f"[Try {i + 1}] Starting serve.run()")
                serve.start(
                    http_options={"host": "0.0.0.0", "port": self.http_port},
                    detached=True, disable_dashboard=os.name == "nt"
                )
                print("✅ serve.start() started successfully")
                break
            except RayActorError as e:
                print(f"⚠️ serve.start() failed, retrying...: {e}")
                time.sleep(2)
            except Exception as e:
                print("🔥 Unexpected error in serve.run():", e)
                time.sleep(2)

    def run_serve(self):

        serve.run(
            HeadServer.options(
                name=self.env_id
            ).bind(),
            route_prefix=f"/{self.env_id}"
        )
        print("✅ serve.run() started successfully")

    def stop(self):
        ray.shutdown()
        print("🛑 ray shutdown")


    def status(self):
        subprocess.run(["ray", "status"])
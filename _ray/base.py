import os
import socket
import subprocess

import ray
from ray import serve

from bm.settings import OS_NAME
from containers.head.main import HeadServer


class RayAdminBase:

    def __init__(self, env_id):
        self.logs_dir = r"C:\Users\wired\OneDrive\Desktop\BestBrain\tmp\ray\session_*\logs" if OS_NAME == "nt" else "/tmp/_ray/session_*/logs"
        self.include_dashboard=OS_NAME!="nt"
        self.local_mode = OS_NAME=="nt"
        self.ip = socket.gethostbyname(socket.gethostname())
        self.ray_port = 6379
        self.http_port = 8001
        self.env_id = env_id
        print("RayBase initialized")

    def start(self):
        os.environ["RAY_DISABLE_DASHBOARD"] = "1" if OS_NAME =="nt" else "0"
        ray.init(
            ignore_reinit_error=True,
            local_mode=self.local_mode,
            include_dashboard=self.include_dashboard,
            address=f"{self.ip}:{self.ray_port}",
        )
        print("ray initialized")

    def start_head(self):
        subprocess.run(["ray", "start", "--head", f"--port={self.ray_port}", "--include-dashboard=false"], check=True)

    def stop_ray(self):
        subprocess.run(["ray", "stop", "--force"], check=True)

    def run_serve(self):
        serve.start(
            http_options={"host": self.ip, "port": self.http_port},
            detached=True, disable_dashboard=os.name == "nt"
        )

        serve.run(
            HeadServer.options(name=self.env_id).bind(),
            route_prefix=f"/{self.env_id}"
        )


    def stop(self):
        ray.shutdown()
        print("_ray shutdown")
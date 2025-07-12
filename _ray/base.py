import os

import ray

from bm.settings import OS_NAME


class RayAdminBase:

    def __init__(self):
        self.logs_dir = r"C:\Users\wired\OneDrive\Desktop\BestBrain\tmp\ray\session_*\logs" if OS_NAME == "nt" else "/tmp/_ray/session_*/logs"
        self.include_dashboard=OS_NAME!="nt"
        self.local_mode = OS_NAME=="nt"
        print("RayBase initialized")

    def start(self):
        os.environ["RAY_DISABLE_DASHBOARD"] = "1" if OS_NAME =="nt" else "0"
        ray.init(
            ignore_reinit_error=True,
            local_mode=self.local_mode,
            include_dashboard=self.include_dashboard,
        )
        print("ray initialized")

    def stop(self):
        ray.shutdown()
        print("_ray shutdown")
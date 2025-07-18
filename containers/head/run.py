import shutil
import os
import socket
import subprocess
import time
import ray
from ray import serve
from ray.exceptions import RayActorError

from _ray.base import RayAdminBase
from containers.head import ENV_ID



if __name__ == "__main__":
    try:
        os.path.expanduser("~")

        ray_admin = RayAdminBase(env_id=ENV_ID)
        ray_admin.stop_ray()

        # → manuelles Löschen alter Ray-Session
        if os.name == "nt":
            ray_tmp_path = os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "Temp", "ray")
            shutil.rmtree(ray_tmp_path, ignore_errors=True)

        ray_admin.start_head()

        os.environ["RAY_LOG_TO_STDERR"] = "1"
        os.environ["RAY_DISABLE_AUTO_WORKER_SETUP"] = "1"

        #time.sleep(5)



        #time.sleep(2)  # kleine Verzögerung
        # Retry serve.run()

        ray_admin.start_serve()

        ray_admin.run_serve()



    except Exception as e:
        print(f"Fehler: {e}")
        subprocess.run(["ray", "stop", "--force"])

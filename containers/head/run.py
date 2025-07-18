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

        for _ in range(10):
            try:
                ray_admin.start()
                break
            except Exception as e:
                print("Retrying ray.init()...")
                time.sleep(1)

        #time.sleep(2)  # kleine Verzögerung
        # Retry serve.run()
        for i in range(10):
            try:
                print(f"[Try {i + 1}] Starting serve.run()")
                ray_admin.run_serve()

                print("✅ serve.run() started successfully")
                break
            except RayActorError as e:
                print("⚠️ serve.run() failed, retrying...")
                time.sleep(2)
            except Exception as e:
                print("🔥 Unexpected error in serve.run():", e)
                time.sleep(2)

    except Exception as e:
        print(f"Fehler: {e}")
        subprocess.run(["ray", "stop", "--force"])

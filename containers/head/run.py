import shutil
import os
import subprocess
import time
import ray
from ray import serve
from ray.exceptions import RayActorError

from containers.head import ENV_ID
from containers.head.main import HeadDepl



if __name__ == "__main__":
    try:
        os.path.expanduser("~")

        subprocess.run(["ray", "stop", "--force"], check=True)
        # → manuelles Löschen alter Ray-Session
        ray_tmp_path = os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "Temp", "ray")
        shutil.rmtree(ray_tmp_path, ignore_errors=True)

        subprocess.run(["ray", "start", "--head", "--port=6379", "--include-dashboard=false"], check=True)
        os.environ["RAY_LOG_TO_STDERR"] = "1"
        os.environ["RAY_DISABLE_AUTO_WORKER_SETUP"] = "1"

        #time.sleep(5)

        for _ in range(10):
            try:
                ray.init(address="auto", )
                break
            except Exception as e:
                print("Retrying ray.init()...")
                time.sleep(1)

        serve.start(detached=True, disable_dashboard=True)  # explizit starten
        #time.sleep(2)  # kleine Verzögerung
        # Retry serve.run()
        for i in range(10):
            try:
                print(f"[Try {i + 1}] Starting serve.run()")
                serve.run(
                    HeadDepl.options(name=ENV_ID).bind(),
                    route_prefix=f"/{ENV_ID}",
                )
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

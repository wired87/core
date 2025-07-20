import shutil

###

import os

from fastapi import FastAPI
from bm.settings import TEST_ENV_ID, TEST_USER_ID
from container.ray_base import RayAdminBase

app = FastAPI()
USER_ID = os.environ.get("USER_ID", TEST_USER_ID)
ENV_ID = os.environ.get("ENV_ID", TEST_ENV_ID)

if __name__ == "__main__":
    ray_admin = RayAdminBase(env_id=ENV_ID)
    try:
        os.path.expanduser("~")
        ray_admin.stop_ray()
        # → manuelles Löschen alter Ray-Session
        if os.name == "nt":
            ray_tmp_path = os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "Temp", "ray")
            shutil.rmtree(ray_tmp_path, ignore_errors=True)

        ray_admin.start_head()

        os.environ["RAY_LOG_TO_STDERR"] = "1"
        os.environ["RAY_DISABLE_AUTO_WORKER_SETUP"] = "1"

        ray_admin.start_serve()

        ray_admin.run_serve()

    except Exception as e:
        print(f"Fehler: {e}")
        ray_admin.stop_ray()

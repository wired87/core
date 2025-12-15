import logging

import ray
import uvicorn
from fastapi import Body

from app_utils import APP, FB_DB_ROOT, GLOBAC_STORE
from fb_core.real_time_database import FBRTDBMgr


@APP.post("/")
async def root_pod():
    print("root pod request...")
    return {"message": "hi", "status": "success"}


@APP.post("/root/")
async def root_cluster(payload: dict = Body(...)):
    print(f"Received POST payload: {payload}")
    if not GLOBAC_STORE.get("HEAD", None):
        print("Sim not ready now")
        return {"status": "error", "msg": "No Head attached"}
    try:
        gloal_state:dict = ray.get(GLOBAC_STORE["DB_WORKER"].get_global_state.remote(), timeout=5)
        # Just check for ready initialized
        ready = gloal_state["ready"]
        if ready is True:
            resp = ray.get(
                GLOBAC_STORE["HEAD"].handle_extern_message.remote(payload, gloal_state)
            )
            return {"status": "success", "data": resp}
        else:
            print("Request failed")
            return {"status": "success", "data": {"message": "Sim not ready now"}}
    except Exception as e:
        print(f"Error while forwarding to Head: {e}")
        return {"status": "error", "msg": str(e)}

class Server:
    """
    Wrapper for FastAPI lifecycle.
    - Start/stop server
    - Handle shutdown cleanly
    - Encapsulate routes so they can interact with Head logic
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("fastapi.server")

    def run(self):
        """Blocking call to run FastAPI via Uvicorn."""
        print(f"Starting FastAPI server on {self.host}:{self.port}")
        uvicorn.run(APP, host=self.host, port=self.port)

    async def run_async(self):
        """Non-blocking run inside an asyncio loop."""
        config = uvicorn.Config(APP, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        print("Starting FastAPI server (async mode)...")
        await server.serve()

if __name__ == "__main__":
    db_manager = FBRTDBMgr()

    globs = {}
    global_states = db_manager.get_data(
        path=f"{FB_DB_ROOT}/global_states/",
    )

    for k, v in global_states.items():
        print(v)
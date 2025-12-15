import asyncio

import ray
from _ray_core.base.base import BaseActor

from app_utils import FB_DB_ROOT

from _ray_core.globacs.state_handler.main import StateHandler

import dotenv

dotenv.load_dotenv()

@ray.remote
class StateHandlerWorker(
    StateHandler,
    BaseActor
):
    def __init__(self, host):
        self.run = False
        self.ready = False
        self.host = host
        BaseActor.__init__(self)

        StateHandler.__init__(self, self.run)

        self.database = FB_DB_ROOT

        self.shutdown_nodes = {}
        print("StateHandler initialized")

    def handle_initialized(self, host):  # set_refs
        self.host.update(host)
        StateHandler.__init__(self, self.run)

    def ping(self):
        return True




    async def main(self, id_map, parent_id):
        print("Start StateHandler.main")
        try:
            await self.set_run(True)
            while self.run is True:
                self.monitor_state(
                    id_map,
                    parent_id
                )
                await asyncio.sleep(5)

        except Exception as e:
            print(f"Exception while start state main: {e}")
        print("State monitoring finished")

    async def set_run(self, run):
        print("Set sh run =", run)
        self.run = run

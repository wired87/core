import asyncio
import ray
from utils.logger import LOGGER


@ray.remote
class StateHandler:

    """
    Compares len active nodes against len all nodes in the cluster.
    If check_state returns True, the sim can be started
    """

    def __init__(self, head_ref):
        self.head_ref = head_ref

    async def check_state(self):
        building = True
        while building:
            active_workers = self.head_ref.get_ative_workers.remote()
            len_active_workers = len(active_workers)
            len_ray_nodes = len(ray.nodes())
            if len_active_workers == (len_ray_nodes - 1):  # -1 becaus head
                building = False
            else:
                LOGGER.info(f"{len_active_workers}/{len_ray_nodes} activated")
                await asyncio.sleep(3)

        self.head_ref.handle_all_workers_active.remote()
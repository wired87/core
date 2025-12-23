import ray
from ray import get_actor

from core._ray_core.base._ray_utils import RayUtils


class BaseActor(RayUtils):

    def __init__(self, parent_id=None):
        RayUtils.__init__(self)
        self.parent_id = parent_id
        self.host = {}

    def ping(self):
        return True

    def handle_initialized(self, host):
        try:
            self.host.update(host)
            if self.parent_id:
                print(f"{self.parent_id} Received host")
        except Exception as e:
            print(f"Error updating host: {e}")


    def get_G(self):
        # restore actor ahndles
        G = ray.get(get_actor(name="UTILS_WORKER").get_data_state_G.remote())
        return G




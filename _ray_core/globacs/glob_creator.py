import ray

from _ray_core.base.base import BaseActor
from _ray_core.globacs.G import UtilsWorker
from _ray_core.globacs.logger_worker import LoggerWorker
from _ray_core.globacs.head import Head
from _ray_core.globacs.state_handler.main import StateHandler
from _ray_core.globacs.state_handler.state_handler import StateHandlerWorker


@ray.remote
class GlobsMaster(BaseActor):

    def __init__(self, world_cfg):
        BaseActor.__init__(self)
        self.world_cfg = world_cfg
        self.alive_workers = []
        self.available_actors = {
            "UTILS_WORKER": self.create_utils_worker, # need first
            "HEAD": self.create_head_server,
            "GLOB_LOGGER": self.create_global_logger,
            "GLOB_STATE_HANDLER": self.create_global_state_handler,
        }

        self.sh = StateHandler(
            run=True
        )

        print("GlobsMaster initiaized")

    def create(self):
        try:
            self.create_globs()
            id_map = list(self.available_actors.keys())
            self.sh.await_alive(
                id_map=id_map
            )
            print("Exit GlobsMaster...")
        except Exception as e:
            print(f"Err GlobCreator.create: {e}")

    def create_globs(self):
        print(f"Create GLOBS")
        retry = 3
        for name in list(self.available_actors.keys()):
            for i in range(retry):
                print(f"Create: {name} \nTry: {i}")
                try:
                    ref = self.available_actors[name](name)

                    self.sh.await_alive(["UTILS_WORKER"])

                    ray.get_actor(name="UTILS_WORKER").set_node.remote(
                        dict(
                            nid=name,
                            ref=ref._ray_actor_id.binary().hex(),
                            type="ACTOR"
                        )
                    )
                    break
                except Exception as e:
                    print(f"Err create_glob {name}: {e}")
        print("GLOB creation finished")

    def create_utils_worker(self, name):
        ref = UtilsWorker.options(
            name=name,
            lifetime="detached",
        ).remote(
            world_cfg=self.world_cfg
        )
        self.sh.await_alive(["UTILS_WORKER"])
        #ray.get(ref.create_world.remote())
        return ref

    def create_global_logger(self, name):
        ref = LoggerWorker.options(
            name=name,
            lifetime="detached"
        ).remote(
            host=self.host
        )
        return ref


    def create_global_state_handler(self, name):
        ref = StateHandlerWorker.options(
            name=name,
            lifetime="detached",
        ).remote(
            host=self.host,
        )
        return ref


    def create_head_server(
            self,
            name
    ):
        ref = Head.options(
            name=name,
            lifetime="detached",
        ).remote()
        print("✅ Head started successfully")
        return ref


if __name__ == "__main__":
    ref = GlobsMaster.remote()

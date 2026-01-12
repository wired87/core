"""import asyncio

import ray
from ray._private.custom_types import TypeActorStatus
from ray.util.state.common import ActorState, WorkerState
from cluster_nodes.cluster_utils.global_actor_utils import GlobalActorUtils

import dotenv
dotenv.load_dotenv()

@ray.remote
class GlobalStateHandler(
    GlobalActorUtils
):

    def __init__(self, host={}):
        self.host = host
        GlobalActorUtils.__init__(self, self.host)

        self.run = False
        self.ready = False

        self.stim_cfg = None

        self.upsert_endpoint = "/metadata"

        self.alive_actors = {}
        self.dead_actors = {}
        self.shutdown_nodes = {}
        print("GlobalStateHandler initialized")

    async def receive_stim_cfg(self, stim_cfg):
        self.stim_cfg = stim_cfg
        if self.all_workers_alive():
            print("Received stim cfg")

    async def distribute_info(self, worker_ids):

        print("Activate runtime utils")
        try:
            self.host["global_logger"].main.remote(
                worker_ids=worker_ids
            )
            await self.distribute_worker_stim_cfg()
            print("Runtime utils activated")
        except Exception as e:
            print(f"Exception while activate runtime utils: {e}")


    async def distribute_worker_stim_cfg(self):
        # Distribute Stim Cfg
        all_subs = ray.get(GLOBAC_STORE["UTILS_WORKER"].get_all_refs.remote(
            just_subs=True,
        ))
        for nid in list(all_subs.keys()):
            if nid in self.stim_cfg:
                print(f"Distributing stim cfg to {nid}")
                ref = ray.get_actor(name=f"{nid}_updator")
                ref.receive_stim_cfg.remote(self.stim_cfg)
            else:
                print(f"Skipping {nid} as no stim cfg found")
        print("Finished Stim cfg distribution")



    async def start(self, monitor_ids):
        print("Start GlobalStateHandler")
        self.monitor_names = monitor_ids
        self.run = True
        self.monitor_state(self.monitor_names)

    def monitor_state(self, monitor_names):
        print("begin monitoring state")
        try:
            # Monitor state of
            while self.run is True:
                for nid, attrs in ray.get(
                        GLOBAC_STORE["UTILS_WORKER"].get_ray_node_infos.remote(
                            id_map=monitor_names
                        )).items():
                    try:
                        actor:ActorState = attrs["actor"]
                        worker:WorkerState = attrs["worker"]
                        #print(f"Check workers {actor.name} state")
                        state: TypeActorStatus = actor.state
                        name = actor.name
                        if name in self.monitor_names:
                            if state == "ALIVE":
                                self.handle_alive_actor(
                                    name,
                                    state,
                                    actor,
                                )

                            elif state == "DEAD":
                                self.handle_dead_actor(
                                    name,
                                    state,
                                    actor,
                                    worker,
                                )
                    except Exception as e:
                        print(f"Error detected while handling state: {e}:")
            else:
                print("State handling finished")
        except Exception as e:
            print("Error runninig GlobStateHandler: ", e)

    def all_workers_alive(self):
        return len(self.alive_actors) >= len(self.monitor_names) and self.ready is False


    def handel_global_state_true(self):
        print("Handle all PXs ALIVE")
        asyncio.run(self.distribute_info(
            worker_ids=list(
                self.alive_actors.keys()
            )
        ))
        if self.all_workers_alive():
            # Upsert ready
            self.ready = True
            self.distribute_state()
            print("All Pixel Nodes ALIVE")
        else:
            print(f"ALIVE: {len(self.alive_actors)}/{len(self.monitor_names)}")


    def handle_alive_actor(
            self,
            name,
            state,
            actor,
    ):
        if name not in self.alive_actors:
            print("Alive actor detected")
            meta = {
                "status": {
                    "state": state,
                    "info": "null",
                },
                "pid": actor.pid,
                "node_id": actor.node_id,
                "class_name": actor.class_name
            }
            print(f"Worker {name} is {state}")

            if name in self.dead_actors:
                self.dead_actors.pop(name)

            self.alive_actors[name] = actor

            # Upsdrt name
            self.host["DB_WORKER"].iter_upsert.remote(
                attrs=meta,
                path=f"{self.database}{self.upsert_endpoint}/{name}"
            )

            print(f"Meta for {name} sent")
            self.handel_global_state_true()

    def handle_dead_actor(
            self,
            name,
            state,
            actor,
            worker,
    ):
        if name not in self.dead_actors:
            print("Dead actor detected")
            worker_details = self.get_worker_details(worker)
            meta = {
                "status": {
                    "state": state,
                    "info": worker_details,
                },
                "pid": actor.pid,
                "node_id": actor.node_id,
                "class_name": worker.class_name
            }

            if name in self.alive_actors:
                self.alive_actors.pop(name)
            self.dead_actors[name] = actor
            # Upsert name
            self.host["DB_WORKER"].iter_upsert.remote(
                attrs=meta,
                path=f"{self.database}/metadata/{name}/"
            )
            print(f"Metadata state for {name} sent")


    def get_worker_details(self, worker):
        worker_data = {
            "exit_detail": worker.exit_detail,
            "exit_type": worker.exit_type,
            "worker_launch_time_ms": worker.worker_launch_time_ms,
            "start_time_ms": worker.start_time_ms,
            "end_time_ms": worker.end_time_ms,
            "is_alive": worker.is_alive,
        }

        #todo logger worker id
        return worker_data

    def distribute_state(self):
        #todo utils worker als spanner stimulator
        try:
            if self.state_handler_type == "HEAD":
                for pixel_id in self.monitor_names:
                    try:
                        self.host["DB_WORKER"].set_ready.remote(ready=self.ready)
                    except Exception as e:
                        print(f"Error setting global state: {e}")
            else:
                print("upsert state for pixel")
                meta = {
                    "status": {
                        "state": "ALIVE",
                        "info": "null"
                    },
                    "pid": "hidden",
                    "node_id": "hidden",
                    "class_name": "PixelWorker"
                }

                # Upsert Pixel state DB
                self.host["DB_WORKER"].iter_upsert.remote(
                    attrs=meta,
                    path=f"{self.database}/metadata/{self.parent_pixel_id}/"
                )

                # Register Active Pixel @ head
                self.host["HEAD"].handle_px_states.remote(
                    px_id=self.parent_pixel_id
                )

        except Exception as e:
            print(f"Error distribute state: {e}")
"""
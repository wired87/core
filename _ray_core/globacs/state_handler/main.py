import ray
from ray._private.custom_types import ACTOR_STATUS
from ray.util.state import list_actors
from ray.util.state.common import ActorState, WorkerState

import dotenv
dotenv.load_dotenv()

from _ray_core.globacs.state_handler.sh_utils import StateHandlerUtils
from core.app_utils import FB_DB_ROOT


class StateHandler(
    StateHandlerUtils
):
    """
    Acts as State Handler for all PXs in a namespace
    """

    def __init__(
            self,
            run=False,
    ):
        #print("StateHandler.host")
        StateHandlerUtils.__init__(self, run)
        self.run = False
        self.ready = False
        self.stim_cfg = None

        self.database = FB_DB_ROOT

        self.upsert_endpoint = "/metadata"

        self.alive_actors = list()
        self.dead_actors = list()

        self.shutdown_nodes = {}

        print("GlobalStateHandler initialized")

    def check_all_alive(self, id_map) -> bool:
        ready = True
        for nid in id_map:
            if nid not in self.alive_actors:
                ready = False
        print("Alive check:", ready)
        return ready


    def receive_refs(self, id_map):
        #print(f"receive_rfs id_map: {id_map}")
        refs = {}
        for name in id_map:
            refs[name] = ray.get_actor(
                name=name
            )
        print("Refs extracted:", refs)
        return refs


    def monitor_state(
            self,
            id_map: list[str]=None,
            parent_id: str = None,
            alive_action=None,
            dead_action=None,
            alive_kwargs=None,
            dead_kwargs=None,
    ):
        # gets all actors per default
        node_info: dict = self.get_ray_node_infos()# id_map=id_map

        if id_map:
            len_keys = len(id_map)
        #print(f"Start monitor state of {len(node_info)} workers")
        for nid, ninfo_struct in node_info.items():
            #print(f"<WORKING {nid} =============>")
            if nid in self.alive_actors:
                continue

            actor: ActorState = ninfo_struct["actor"]
            worker: WorkerState = ninfo_struct["worker"]
            # print(f"Check workers {actor.name} state")
            state = actor.state
            #print(f"stste: {state}")

            if state == "ALIVE":
                self.handle_alive(
                    state,
                    actor,
                    actor.name,
                )
                print(f"{len(self.alive_actors)}/{len(node_info)} entries ALIVE")

            elif state == "DEAD":
                self.handle_dead_actor(
                    actor.name,
                    state,
                    actor,
                    worker,
                )
                print(f"{len(self.dead_actors)}/{len(node_info)} entries DEAD")

            if len(self.dead_actors):
                print(f"[STATE_HANDLER] {parent_id}:")
                for nid in self.dead_actors:
                    print(f">{nid}")

            if len(self.alive_actors) == len(node_info.keys()):
                print("all actors alive")
        return True


    async def distribute_worker_stim_cfg(self):
        # Distribute Stim Cfg
        all_subs = ray.get()
        for nid in list(all_subs.keys()):
            if nid in self.stim_cfg:
                print(f"Distributing stim cfg to {nid}")
                ref = ray.get_actor(name=f"{nid}_updator")
                ref.receive_stim_cfg.remote(self.stim_cfg)
            else:
                print(f"Skipping {nid} as no stim cfg found")
        print("Finished Stim cfg distribution")

    def handle_dead_actor(
            self,
            name,
            state,
            actor,
            worker,
    ):
        try:
            if name in self.dead_actors:
                return
            elif name in self.alive_actors:
                self.alive_actors.remove(name)
            #print(f"Dead actor detected: {name}")
            self.dead_actors.append(name)

            worker_details = self.get_worker_details(worker)
            meta = {
                "status": {
                    "state": state,
                    "info": worker_details,
                },
                "pid": actor.pid,
                "node_id": actor.node_id,
                "class_name": actor.class_name
            }

            # Upsert name
            """
            #GLOBAC_STORE["DB_WORKER"].iter_upsert.remote(
                attrs=meta,
                path=f"{self.database}/metadata/{name}/"
            )
            """
            #print(f"Metadata state for {name} sent")
        except Exception as e:
            print(f"Err handling DEAD state: {e}")


    def handle_alive(
            self,
            state,
            actor,
            name,
    ):
        try:
            if "HEAD" in name:
                name = "HEAD"

            if "SERVE_REPLICA" in name:
                name = name.split("::")[1].split("#")[0]

            if name in self.alive_actors:
                print(f"Skipping alive actor {name}")
                return

            elif name in self.dead_actors:
                self.dead_actors.remove(name)

            self.alive_actors.append(name)

            if "SERVE_PROXY_ACTOR" in name:
                return

            print(f"Upsert state for {name}")
            self.state_upsert(
                state,
                actor,
                name,
            )
        except Exception as e:
            print(f"Err handling ALIVE state: {e}")


    def state_upsert(
            self,
            state,
            actor,
            name,
    ):
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

        print(f"Meta for {name} upserted")

    def await_alive(
            self,
            id_map:list[str],
    ):
        """
        Wait for all actors and deployments in id_map to be alive.
        
        Handles both:
        - Actors: Checks via list_actors()
        - Deployments: Checks via Ray Serve API (deployments stored as "DEPLOYMENT_{name}" in graph)
        
        Args:
            id_map: List of actor/deployment names to wait for. 
                   Deployment names should match the deployment_name (not the "DEPLOYMENT_" prefix)
        """
        
        alive_actors = set()
        alive_deployments = set()
        total_expected = len(id_map)
        total_alive = 0
        
        print(f"Awaiting {len(id_map)} actors: {id_map}")
        
        while total_alive < total_expected:
            # Check actors
            if len(id_map) > 0:
                all_actors = list_actors(detail=True)
                for actor in all_actors:
                    try:
                        state: ACTOR_STATUS = actor.state
                        aname = actor.name
                        if aname in id_map:
                            print(f"{aname}: {state}")
                            if ("REPLICA" not in aname):
                                if state == "ALIVE":
                                    if aname not in alive_actors:
                                        alive_actors.add(aname)
                                        total_alive = len(alive_actors) + len(alive_deployments)
                                        print(f"Actor {aname} is ALIVE ({total_alive}/{total_expected} total alive)")
                                elif state == "DEAD":
                                    print(f"Actor {aname} is DEAD")
                    except Exception as e:
                        print(f"Err await_alive checking actor: {e}")
        
        print(f"✅ All {total_alive} handles are alive ({len(alive_actors)} actors, {len(alive_deployments)} deployments)")
        return True

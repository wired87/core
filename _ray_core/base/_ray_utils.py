import os

try:
    import ray
    from ray.actor import ActorHandle
    from ray.util.state import list_actors, list_workers
except ImportError:
    class MockRay:
        class util:
             def list_named_actors(self, all_namespaces=True):
                 return []
        def get_actor(self, *args):
            return None
    ray = MockRay()
    class ActorHandle:
        pass
    def list_actors(detail=True):
        return []
    def list_workers(detail=True):
        return []


class RayUtils:

    def __init__(self):
        self.ray_assets_dir = r"C:\Users\bestb\Desktop\qfs\tmp\ray" if os.name == "nt" else "/tmp/ray/"
        os.makedirs(self.ray_assets_dir, exist_ok=True)
        print("RayUtils initialized")

    def _p(self, msg:str, logger=None):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def list_actors(self, print_actors=False):
        actors = ray.util.list_named_actors(all_namespaces=True)
        if print_actors is True:
            print(f"Aktive Remote-Instanzen: {len(actors)}")
            for actor in actors:
                print(actor)  # Zeigt Name oder Handle
        return actors


    def get_all_actor_refs(self) -> dict[str, ActorHandle]:
        all_actors = list_actors(detail=True)
        refs = {}
        for actor in all_actors:
            aname = actor.name
            if "SERVE_PROXY_ACTOR" not in aname and "SERVE_CONTROLLER_ACTOR" not in aname:
                try:
                    refs[aname] = ray.get_actor(aname)
                except Exception as e:
                    print(f"Err get_all_actor_refs: {e}")
        return refs


    def get_ray_node_infos(self, id_map=None):
        all_actors = list_actors(detail=True)
        all_workers = list_workers(detail=True)

        worker_pid_map = {
            worker.pid: worker
            for worker in all_workers
        }

        struct = {}
        for actor in all_actors:
            # Verarbeite nur Actors, die einen Namen und eine PID haben.
            if actor.name and actor.pid:
                # Finde den passenden Worker.
                corresponding_worker = worker_pid_map.get(actor.pid)

                # Wenn ein passender Worker gefunden wurde...
                if corresponding_worker:
                    if id_map is not None and len(id_map):
                        aname = actor.name
                        # check name
                        if aname in id_map or "HEAD" in aname:
                            #print(f"Include {actor.name}")
                            struct[actor.name] = {
                                "actor": actor,
                                "worker": corresponding_worker
                            }
                    else:
                        struct[actor.name] = {
                            "actor": actor,
                            "worker": corresponding_worker
                        }
        return struct


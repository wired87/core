import ray
from _ray_core.base._ray_utils import RayUtils


class StateHandlerUtils(RayUtils):

    def __init__(self, run,fb_user_root=None):
        super().__init__()
        self.run = run
        self.database=fb_user_root

        self.upsert_endpoint= "/metadata"



    def get_px_struct(self):
        pixel_struct= {
            "DB_WORKER": None,
            "LOGGER": None,
            "state_handler": None,
            "pixel_worker": None
        }

        # ALL PXs
        """all_px = ray.get(GLOBAC_STORE["UTILS_WORKER"].call.remote(
            method_name="get_nodes",
            filter_key="type",
            filter_value=["PIXEL"]
        ))"""

        # CREATE STRUCT
        activate_struct = {
            px_id: pixel_struct
            for px_id in list(all_px.keys())
        }

        # ACTOR NAMES OF INTEREST
        all_listener_names = [
            f"{name}_{px_id}"
            for px_id, struct in activate_struct.items()
            for name in struct.keys()
        ]

        # get ray info of that workers
        return activate_struct, [
            *all_listener_names,
            *list(all_px.keys())
        ], pixel_struct



    def check_refs_exists_list(self, refs:dict):
        all_refs_active = True
        for key, ref in refs.items():
            if ref is None:
                print(f"ref {key} is None")
                all_refs_active = False
        return all_refs_active


    async def distiribute_host(self, host):
        print(f"distiribute_host {host}")
        for name, ref in self.get_all_actor_refs().items(): #host.items():
            if "ServeReplica" in name:
                # Serve App
                await ref.handle_initialized.remote(host=host)
                continue
            ray.get(ref.handle_initialized.remote(host=host))

        print(f"Host distributed")

    def get_worker_details(self, worker):
        worker_data = {
            "exit_detail": worker.exit_detail,
            "exit_type": worker.exit_type,
            "worker_launch_time_ms": worker.worker_launch_time_ms,
            "start_time_ms": worker.start_time_ms,
            "end_time_ms": worker.end_time_ms,
            "is_alive": worker.is_alive,
        }
        # todo logger worker id
        return worker_data



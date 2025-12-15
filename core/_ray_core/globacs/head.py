import os
import time
import ray
import dotenv
dotenv.load_dotenv()

from app_utils import ENV_ID, GLOBAC_STORE
from _ray_core.base.base import BaseActor
from qf_utils.all_subs import ALL_SUBS


@ray.remote(
    num_cpus=.2,
    num_gpus=0,
)
class Head(BaseActor):
    """
    Tasks:
    com with relay & PIXEL
    distribute commands to the right nodes
    wf:
    everything starts
    pixel dockers initialize and msg the HeadDepl (self).
    self message front
    """

    def __init__(self):
        BaseActor.__init__(self)
        print("Initializing HEAD...")

        self.host = {}
        self.creators = {}

        self.receiver = None
        self.attrs = None
        self.external_vm = None
        self.ws_key = None
        self.local_key = None
        self.all_subs = None
        self.all_px_ids = None

        self.shutdown_sub_heads = set()
        self.active_sub_heads = {}
        self.all_px_ready = False
        self.session_id = ENV_ID
        self.node_type = "HEAD"

        self.possible_states = [
            "start",
            "check_status",
            "set_parameters",
            "stop",
        ]

        self.worker_states = {}
        self.extern_host = {}
        self.all_pixels = {}
        self.active_ntypes = {}

        self.messages_sent = 0

        # Listen to DB changes
        self.all_worker_active = False
        print("HeadDeplDeployment initialisiert!")

    async def handle_initialized(self, host):
        print("Head received host")
        try:
            self.host.update(host)
        except Exception as e:
            print(f"Err handle_initialized: {e}")


    async def handle_ntype_state(self, table, refs):
        print(f"======= RECEIVED QStore STATE {table} ========")
        # todo check state abotu received actor
        if table not in self.active_ntypes:
            self.active_ntypes[table] = refs

        await self.check_activate(
            len_all_sub_heads=len(ALL_SUBS)
        )

    async def check_activate(self, len_all_sub_heads: int):
        if len(list(self.active_ntypes.keys())) >= len_all_sub_heads:
            print("================ START SIM PROCESS ==================")
            try:
                # upsert GLOBAL STATE
                self.host["DB_WORKER"].set_ready.remote(ready=True)

                # reset creator refs
                self.creators = {}
                start_time = int(time.time()) + 15

                # Start QStores
                for table, ref_struct in self.active_ntypes.items():
                    for nid, ref in ref_struct.items():
                        ref.init_sim.remote(
                            start_time
                        )
                self.all_px_ready = True
                print("Global state sent")
            except Exception as e:
                print(f"Error setting global state: {e}")



    def all_ready(self):
        return self.all_px_ready

    async def distribute_initialized(self):
        """
        Distirbute trigger to get enighbor refs
        """
        try:
            all_field_workers: list[str] = ray.get(GLOBAC_STORE["UTILS_WORKER"].get_field_worker_ids.remote())
            for nid in all_field_workers:
                print(f"distribute init to {nid}")
                try:
                    ref = ray.get_actor(nid)
                    ray.get(
                        ref.handle_initialized.remote()
                    )
                except Exception as e:
                    print(f"Err distribute_initialized.handle_initialized: {e}")
        except Exception as e:
            print(f"Err distribute_initialized: {e}")

    async def receive_creator(self, nid, ref):
        print(f"Receive Creator ref for {nid}...")
        self.creators[nid] = ref

    async def register_pixel_shutdown(self, px_id):
        try:
            print(f"Register pixel {px_id} to shutdown")
            if self.all_pixels is None:
                self.all_pixels = list(
                    GLOBAC_STORE["UTILS_WORKER"].call.remote(
                        method_name="get_node_list",
                        trgt_types=["PIXEL"]
                    ).keys()
                )

            # Register PX
            if px_id not in self.shutdown_sub_heads:
                self.shutdown_sub_heads.add(px_id)

            if len(self.shutdown_sub_heads) == len(self.all_pixels):
                print("All pixels registered for shutdown -> change global state")
                self.shutdown_sys()

            else:
                print(f"{len(self.shutdown_sub_heads)}/{len(self.all_pixels)} registered")
        except Exception as e:
            print(f"Err register_pixel_shutdown: {e}")


    def reset_env_vars(self):
        try:
            for k, v in os.environ.items():
                os.environ.pop(k)
            print("Env vars reset")
        except Exception as e:
            print(f"FAILED reset env vars: {e}")



    def stop_action(self, all_subs):
        for nid, attrs in all_subs:
            attrs["ref"].exit.remote()
            print(f"Stopped worker {nid}")

    async def handle_all_workers_active(self):
        """
        Whaen everything is init send msg to db
        """
        print("All workers are active, sending ready state to DB")
        self.ready = True

    async def distribute_state_change(self, state, ncfgs):
        print("\n===============DISTRIBUTE STATE=================")
        if ncfgs is None:
            ncfgs = {}
        try:
            # Distribute start
            start_time = int(time.time()) + 10
            payload = {
                "start_time": start_time,
                "type": state,
            }

            print(f"Distribute payload: {payload}")

            # start actors
            try:
                # self.all_subs = all refs of updator field workers
                if len(list(self.all_subs.keys())):
                    print(f"Distribute received states")
                    for i, (nid, ref) in enumerate(self.all_subs.items()):
                        # EXTEND PAYLOAD STRUCT
                        if state == "start" and nid in ncfgs:
                            payload["node_cfg"] = ncfgs[nid]

                        ref.initial_iter_trigger.remote(
                            payload
                        )
                        print(f"{i}/{len(list(self.all_subs.keys()))} states sent")

                    print(f"Finished distribution")

                else:
                    print(f"No refs found. Cant distribute rcvd state")
            except Exception as e:
                print(f"Error while distribute: {e}")
        except Exception as e:
            print(f"Error handling external message {e}")

    def check_ready(self, gloal_state):
        return gloal_state["ready"] and gloal_state["authenticated"] and gloal_state["min_node_cfg_created"]

        # Get ready
        print(f"Changed global_state: {global_state}")
        ready = global_state.get("ready", False)

        if ready is False:
            print("Sim not ready.")
        return ready

    def get_session_id(self):
        return self.session_id

    async def set_ws_validation_key(self, key):
        self.ws_key = key

    async def get_ative_workers(self):
        return self.worker_states["active"]

    async def set_all_subs(self, all_subs):
        if not len(self.all_subs):
            self.all_subs = all_subs
            print("ALL_SUBS set for head")

    def get_actor_info(self):
        struct = {}
        # Get the job id.
        struct["job_id"] = ray.get_runtime_context().get_job_id()
        # Get the actor id.
        struct["actor_id"] = ray.get_runtime_context().get_actor_id()
        # Get the task id.
        struct["task_id"] = ray.get_runtime_context().get_task_id()
        return struct

    async def handle_px_states(self, px_id):
        print("Active Pixel registered")
        if len(list(self.all_pixels.keys())) == 0:
            self.all_pixels: dict = ray.get(
                GLOBAC_STORE["UTILS_WORKER"].call.remote(
                    method_name="get_node_list",
                    trgt_types=["PIXEL"]
                )
            )

        if px_id not in self.active_sub_heads:
            self.active_sub_heads.add(px_id)
        self.check_activate(len(list(self.all_pixels.keys())))

    async def handle_extern_message(self, payload, global_state):
        """
        Handle external messages forwarded by the server.
        
        Example data package to be sent (returned) for payment:
        {
            "type": "payment",
            "state": 1, # 1 for upgrade, -1 for downgrade
            "url": "https://checkout.stripe.com/...", # Stripe URL
        }
        """
        print(f"HEAD handling extern message: {payload}")
        
        try:
            from gem_core.gem import Gem
            gem = Gem()
            
            user_input = payload.get("input", "") or payload.get("message", "")
            
            if not user_input:
                return {"status": "ignored", "msg": "No input provided"}
            
            prompt = f"""
            Classify the following user terminal input into one of these categories:
            - PLAN_UPGRADE: User wants to upgrade their subscription or plan.
            - PLAN_DOWNGRADE: User wants to downgrade their subscription or plan.
            - CAMERA_REQUEST: User wants to open the camera, take a photo, or see themselves.
            - OTHER: Any other request.
            
            Input: "{user_input}"
            
            Return ONLY the category name.
            """
            
            classification = gem.ask(prompt).strip()
            print(f"Classification: {classification}")
            
            if classification == "CAMERA_REQUEST":
                try:
                    import cv2
                    import base64
                    
                    # Open camera
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        return {"status": "error", "msg": "Could not open camera 📷"}
                    
                    # Read frame
                    ret, frame = cap.read()
                    cap.release()
                    
                    if not ret:
                        return {"status": "error", "msg": "Failed to capture image 📷"}
                    
                    # Encode to JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    image_data = f"data:image/jpeg;base64,{jpg_as_text}"
                    
                    return {
                        "type": "camera_capture",
                        "symbol": "📷",
                        "image": image_data,
                        "style": {
                            "borderRadius": "15px"
                        },
                        "target_section": "file_section", # Hint for frontend
                        "msg": "Camera captured successfully 📷"
                    }
                    
                except Exception as e:
                    print(f"Camera error: {e}")
                    return {"status": "error", "msg": f"Camera error: {str(e)} 📷"}

            if classification in ["PLAN_UPGRADE", "PLAN_DOWNGRADE"]:
                state = 1 if classification == "PLAN_UPGRADE" else -1
                
                # TODO: Generate real Stripe URL here
                # For now, we simulate success/failure or just return a mock URL
                stripe_url = "https://checkout.stripe.com/test" 
                success = True # Mock success
                
                if success:
                    return {
                        "type": "payment",
                        "state": state,
                        "url": stripe_url
                    }
                else:
                    # Generate failure message with Gem
                    fail_prompt = "Generate a short, friendly payment failure message with an error icon for a terminal interface."
                    fail_msg = gem.ask(fail_prompt).strip()
                    return {
                        "type": "payment",
                        "state": state,
                        "error": fail_msg
                    }
            
            return {"status": "processed", "classification": classification}
            
        except Exception as e:
            print(f"Error in handle_extern_message: {e}")
            return {"status": "error", "msg": str(e)}

import os
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from core.guard import Guard

"""

[2025-12-19 15:55:39,521 E 3044 30260] rpc_client.h:203: Failed to connect to GCS within 60 seconds. GCS may have been killed. It's either GCS is terminated by `ray stop` or is killed unexpectedly. If it is killed unexpectedly, see the log file gcs_server.out. https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#logging-directory-structure. The program will terminate.
Windows fatal exception: access violation

"""


import requests
import json
import dotenv
import zipfile
import io
import pandas as pd
from channels.generic.websocket import AsyncWebsocketConsumer

import asyncio
from urllib.parse import parse_qs
import importlib

from core.actors import deploy_guard
from qf_utils.qf_utils import QFUtils
from fb_core.real_time_database import FBRTDBMgr

from _chat.log_sum import LogAIExplain
from bm.settings import TEST_ENV_ID
from openai_manager.ask import ask_chat
from visualize import get_convert_bq_table
from workflows.create_ws_prod import WorldCreationWf
from workflows.data_distirbutor import DataDistributor
from workflows.deploy_sim import DeploymentHandler
from workflows.node_cfg_manager import NodeCfgManager
from utils.deserialize import deserialize

from _chat.main import AIChatClassifier

from utils.dj_websocket.handler import ConnectionManager
from utils.get_local_ip import get_local_ip

from utils.graph.local_graph_utils import GUtils
from utils.id_gen import generate_id
from utils.utils import Utils

from fb_core.firestore_manager import FirestoreMgr
from dataclasses import dataclass
from typing import Callable, Any, Awaitable

@dataclass
class RelayCase:
    case: str
    description: str
    callable: Callable[[Any], Awaitable[Any]]

dotenv.load_dotenv()

class Relay(
    AsyncWebsocketConsumer
):
    """
    Komplette lgik ausgelagert auf Ray _qfn_cluster_node
    instanz primär für validierung

    Handlet user requests zum start einer sim
    startet externes GKE _qfn_cluster_node mit sim
    _ray_core env worker muss eine ws sein

    ABER:
    Diese optimierung machst du erst
    nachdem du einen KD hast!!!
    (davor läuft alles auf EINER VM)

    Validates User
    Fetches given env from DB and builds a G from it
    Creates New websocket connections for each QFN and reg. them in the channel-pool (-> new items then do can send messages to all pool items without establish a sepparate connection)

    todo: handles updates (stim,...) and distribute to single nodes

    Testing:
    Keine websocket solang keine extra VM -> gib direkt an die ref weiter

    THERE IS NO RELAY->SERVER COMMUNICATION -> EVERYTHING HAPPENS THROUGH DB LISTENER

    todo connect without env_id & creds -> handle everythng in wold cfg rq
    # todo start docker locally
    todo -> handle node cfg
    # todo check cfgs collected when try to start
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_ips = None
        self.domain = "http://127.0.0.1:8000" if os.name == "nt" else "https://bestbrain.tech"

        self.cluster_ws = None
        self.cluster_acess_ip:int = None

        self.connection_manager = ConnectionManager()
        self.chat_classifier = AIChatClassifier()

        self.utils = Utils()

        # save fiel names to apply to envs
        self.file_store:list[str] = []
        self.qc = False
        self.instance = os.environ["FIREBASE_RTDB"]
        self.demo_g_in_front = False
        self.start_up_path = "container/run.py"
        self.testing=True
        self.ready_sessions = []
        self.sessions_created = []
        self.created_envs = []

        self.env_node = None
        self.sim_state_handler = None
        self.stimulator = None
        self.user_id = None
        self.run = True
        self.database = None
        #self.ray_admin = None
        self.ws_port = None
        self.env_store = []
        self.data_request_endpoint = f"{self.domain}/bq/auth"

        self.fields = []

        self.ws_handler = None
        self.external_vm = False
        self.sim_ready = False
        self.sim_start_puffer = 10  # seconds to wait when rcv start trigger
        self.demo = True

        self.required_steps = {
            "node_cfg": False,
            "world_cfg": False,
            "injection_cfg": False,
        }

        self.active_envs = {}
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)

        self.worker_states = {
            "unknown": [],
            "error": [],
            "inactive": [],
            "active": [],
        }

        self.possible_states = [
            "start",
            "check_status",
            "set_parameters",
            "stop",
        ]

        self.loop = asyncio.new_event_loop()
        self.data_thread_loop = asyncio.new_event_loop()

        self.cluster_auth_data = None

        self.db_manager = FBRTDBMgr()

        self.con_type="http" if os.name == "nt" else "https"
        self.cluster_domain = "127.0.0.1:8001" if os.name == "nt" else "clusterexpress.com"
        self.cluster_url = f"{self.con_type}://{self.cluster_domain}/"
        print(f"Cluster domain set: {self.cluster_url}")

        if self.testing is True:
            self.cluster_root = "http://127.0.0.1:8001"

        else:
            self.cluster_root = "cluster.botworld.cloud"

        self.auth_data = None
        
        self.firestore = FirestoreMgr()

        self.relay_cases: list[RelayCase] = []
        self._register_cases()
        threading.Thread(target=self._scan_dynamic_cases, daemon=True).start()


    async def connect(self):
        print(f"Connection attempt registered")
        # Create ID
        await self.accept()
        print("Scope type:", self.scope.get("type"))
        query_string = self.scope["query_string"].decode()
        query_params = parse_qs(query_string)
        print(f"query_params extracted {query_params}")

        user_id = query_params.get("user_id")[0]  # todo user_id create frontend
        self.user_id = user_id

        self.session_id = generate_id()

        print("self.user_id", self.user_id)

        # todo collect more sim admin_data like len, elements, ...
        # todo improve auth
        if not self.user_id:
            print(f"Connection attempt declined")
            await self.close()
            return

        print(f"Connection attempt verified {self.user_id}")

        self.g = GUtils(
            nx_only=False,
            G=None,
            g_from_path=None,
        )
        self.qfu = QFUtils(
            g=self.g
        )

        self.world_creator = WorldCreationWf(
            user_id=self.user_id,
            parent=self,
            cluster_root=self.cluster_root,
            g=self.g,
            instance=self.instance,
            database=self.database,
            testing=self.testing,
        )

        self.data_distributor = DataDistributor(
            parent=self,
            testing=self.testing,
            user_id=self.user_id,
            db_manager=self.db_manager
        )

        self.node_cfg_manager = NodeCfgManager(
            self.user_id,
            self.db_manager,
            self.cluster_url,
            self.utils,
            parent=self,
        )

        # Create Guard for each new user
        self.guard = Guard(
            self.qfu,
            self.g,
            self.user_id,
        )

        self.deployment_handler = DeploymentHandler(
            user_id
        )

        print("request accepted")

    async def validate_action(self, action_type, data):
        # Validation logic removed, all actions are now permitted
        return True






    def _register_cases(self):
        cases = [
            ("world_cfg", "Process world configuration", self._handle_world_cfg),

            # INJECTION HANDLER
            ("node_cfg", "Process node configuration", self._handle_node_cfg),

            # module handler -> include in bar receive
            ("file", "Handle file uploads", self._handle_files),

            ("request_inj_screen", "Requesting admin_data relavant for inj setup (fields, grid conc etc)",
             self.request_inj_process_start),

            ("start_sim", "Start simulation", self._handle_start_sim_wrapper),

            ("set_inj_pattern", "Set ncfg injection pattern", self.set_cfg_process),

            ("env_ids", "Retrieve environment IDs", self._handle_env_ids),
            ("get_data", "Fetch and zip admin_data from BigQuery", self._handle_get_data),
            ("delete_env", "Delete an environment", self._handle_delete_env),
            ("extend_gnn", "Extend GNN", self._handle_extend_gnn),
            # train is performed in each sim
            #("train_gnn", "Train GNN", self._handle_train_gnn),

            ("create_visuals", "Create visuals", self._handle_create_visuals),
            ("create_knowledge_graph_from_data_tables", "Create KG from admin_data tables", self._handle_create_kg),
        ]
        
        for case, desc, func in cases:
            self.relay_cases.append(RelayCase(case=case, description=desc, callable=func))




    def _scan_dynamic_cases(self):
        """
        Scans environment variables starting with 'RELAY' for paths to dictionaries
        mapping case names to callables.
        Format of env var value: 'module.path.variable_name'
        """
        for key, value in os.environ.items():
            if key.startswith("RELAY"):
                try:
                    # Assume value is "module.path.dict_name"
                    if "." in value:
                        module_name, dict_name = value.rsplit(".", 1)
                        module = importlib.import_module(module_name)
                        case_dict = getattr(module, dict_name)
                        
                        if isinstance(case_dict, dict):
                            print(f"Loading dynamic cases from {key} -> {value}")
                            for case_name, handler in case_dict.items():
                                if callable(handler):
                                    # Check if case already exists to avoid duplicates/overwrites if desired, 
                                    # but typically dynamic overwrites static or appends.
                                    # Here we append.
                                    self.relay_cases.append(
                                        RelayCase(
                                            case=case_name,
                                            description=f"Dynamic case from {key}",
                                            callable=handler
                                        )
                                    )
                                else:
                                    print(f"Skipping non-callable handler for case {case_name} in {key}")
                except Exception as e:
                    print(f"Failed to load dynamic cases from {key} ({value}): {e}")

    async def receive(
            self,
            text_data=None,
            bytes_data=None
    ):
        print(f"start receive: {text_data}")
        try:
            data = deserialize(text_data)
            data_type = data.get("type")  # assuming 'type' field for command
            print(f"Received message from frontend: {data}")

            # Middleware Guard
            if data_type in [
                "world_cfg", "start_sim"
            ] and not self.testing:
                if not await self.validate_action(data_type, data):
                    return

            # Dynamic Dispatch
            handled = False

            # Iterate over a copy to ensure thread safety if cases are added at runtime
            for relay_case in list(self.relay_cases):
                if relay_case.case == data_type:
                    await relay_case.callable(data)
                    handled = True
                    break
            
            if not handled:
                print(f"Unknown command type received: {data_type}")

        except Exception as e:
            print(f">>Error processing received message: {e}")
            import traceback
            traceback.print_exc()


    async def _handle_world_cfg(self, data: dict):
        try:
            print("CREATE WORLD REQUEST RECEIVED", data)
            world_cfg = data["world_cfg"]
            # Frontend sends world_cfg as a list, extract first element
            if isinstance(world_cfg, list) and len(world_cfg) > 0:
                world_cfg = world_cfg[0]
            
            # Normalize field names: frontend uses 'amount_of_nodes', Guard expects 'amount_nodes'
            if "amount_of_nodes" in world_cfg and "amount_nodes" not in world_cfg:
                world_cfg["amount_nodes"] = world_cfg["amount_of_nodes"]
            
            # Ensure env_id is present (frontend sends 'id')
            if "env_id" not in world_cfg and "id" in world_cfg:
                world_cfg["env_id"] = world_cfg["id"]
            
            node = self.guard.set_wcfg(world_cfg)
            self.required_steps["world_cfg"] = True
            await self.send(
                text_data=json.dumps({
                    "type": "world_cfg",
                    "world_cfg": node,
                })
            )
            print("world cfg set")
        except Exception as e:
            print(f"Err _handle_world_cfg: {e}")


    async def _handle_node_cfg(self, data: dict):
        print("CREATE NODE CFG REQUEST RECEIVED")
        self.world_creator.node_cfg_process(data)



    async def _handle_env_ids(self):
        await self.send(
            text_data=json.dumps({
                "type": "env_ids",
                "admin_data": self.db_manager.get_child(
                    path=f"users/{self.user_id}/env/"
                ),
            })
        )


    async def request_inj_process_start(self, data):
        env_id=data["env_id"]
        # todo return admin_data for the interactive 3d cube
        if not await self.guard.get_state():
            await self.send(
                text_data=json.dumps({
                    "type": "inj_pattern_struct_err",
                    "admin_data": "You must set world cfg and node cfg fields before set patterns for them... ",
                    "env_id": env_id,
                })
            )
        else:
            data_struct = await self.guard.get_inj_pattern_data()
            print("admin_data for inj init set")
            await self.send(
                text_data=json.dumps({
                    "type": "inj_pattern_struct",
                    "admin_data": data_struct,
                    "env_id": env_id,
                })
            )


    async def set_cfg_process(self, data):
        print("inj_cfg_process start")
        # get guard of user
        env_id = data.get("env_id")
        if env_id:
            self.guard.set_inj_pattern(
                data.get("inj_pattern") or data.get("inj_pattern"),
                env_id
            )
        print("inj_cfg_process cfg set")



    async def _handle_get_data(self, data: dict):
        env_id = data.get("env_id")

        MOCK_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "bigquery-public-admin_data")
        MOCK_CREDENTIALS_PATH = r"C:\Users\bestb\PycharmProjects\BestBrain\auth\credentials.json" if os.name == "nt" else "auth/credentials.json"

        # Get Data from BQ
        csv_data = get_convert_bq_table(
            project_id=MOCK_PROJECT_ID,
            dataset_id="QCOMPS",
            table_id=env_id,
            credentials_file=MOCK_CREDENTIALS_PATH
        )
        
        if isinstance(csv_data, pd.DataFrame):
            # Clean up double header if present (visualize.py artifact)
            if not csv_data.empty and csv_data.iloc[0].tolist() == csv_data.columns.tolist():
                csv_data = csv_data.iloc[1:]

            # Create ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                chunk_size = 10000
                total_rows = len(csv_data)
                
                if total_rows == 0:
                        zip_file.writestr("data_empty.csv", "")
                else:
                    for i in range(0, total_rows, chunk_size):
                        chunk = csv_data.iloc[i:i+chunk_size]
                        csv_string = chunk.to_csv(index=False)
                        zip_file.writestr(f"data_part_{i//chunk_size}.csv", csv_string)
            
            zip_buffer.seek(0)
            print(f"Sending zip file of size: {len(zip_buffer.getvalue())} bytes")
            await self.send(bytes_data=zip_buffer.getvalue())

        else:
            print(f"csv_data received (error): {csv_data}")
            # send error to front
            await self.send(
                text_data=json.dumps({
                    "type": "get_data_error",
                    "admin_data": str(csv_data)
                })
            )

    async def _handle_files(self, data):

        ### todo impl, cachiin (bq->byes + embed -> ss -> if exists: get id(name) -> save local; else: self.handle fiels incl embeds -> bq

        # HANDLE FILES
        files = data.get("files")
        if files:
            # hande files -> save G
            await self._handle_files(files)

        # todo upser files

        self.guard.handle_mod_stack(
            files,
            self.root
        )

        await self.send(
            text_data=json.dumps({
                "type": "message",
                "admin_data": "Messages processed successfully",
            })
        )

    async def _handle_delete_env(self, data: dict):
        env_id = data.get("env_id")
        self.db_manager.delete_data(
            path=f"users/{self.user_id}/env/{env_id}"
        )
        try:
            self.created_envs.remove(env_id)
        except ValueError:
            pass
        await self.send(
            text_data=json.dumps({
                "type": "delete_env",
                "admin_data": f"Deleted {env_id} succsssfully",
            })
        )

    async def _handle_extend_gnn(self, data: dict):
        # extend a gnn with
        pass

    async def _handle_train_gnn(self, data: dict):
        """
        get nv ids fetch admin_data and train a gan e.g.
        """
        pass

    async def _handle_create_visuals(self, data: dict):
        """
        Fetch ad create visuals for a single env.
        The requested anmation gets returned in mp4 format (use visualizer)
        """
        # create expensive id map
        # -> fetch rows for each px t=0
        # sleep.1
        # restart ->
        pass

    async def _handle_create_kg(self, data: dict):
        env_ids:list[str] = data.get("env_ids")
        """
        create nx from all envs
        embed 
        langchain grag
        store local fro query 
        """
        pass

    async def _handle_start_sim_wrapper(self, data: dict):
        # APPLY COLLECTED FILE NAMES TO ENVS
        """
        for env_id in self.created_envs:
            path = f"users/{self.user_id}/env/{env_id}"
            self.db_manager.upsert_data(
                path=path,
                admin_data={"files": self.file_store},
            )
        """
        # todo thread
        env_ids = data.get("env_ids")
        self.guard.main(env_ids)



    def handle_files(self, files):
        if len(files):
            for f in files:
                f_bytes = f.read()
                name = ask_chat(
                    prompt="Create a unique name for the provided file",
                    base64_string=f_bytes
                )

                #

                #self.file_store.append(name)

                """
                self.db_manager.upsert_data(
                    path=f"{self.user_id}/files",
                    admin_data={
                        name: base64.b64encode(f_bytes).decode("utf-8")
                    }
                )
                """
                print("Uploaded file:", name)


    async def batch_inject_env(self, data):
        print("START SIM REQUEST RECEIVED")
        
        envs = self.g.get_nodes_by_type(
            filter_key="ENV",
            filter_value=self.user_id
        )

        for env in envs:
            env_id = env["nid"]
            try:
                if self.world_creator.env_id_map:

                    self.deployment_handler.create_vm(
                        instance_name=env_id,
                        testing=self.testing,
                    )

                    await self.send(
                        text_data=json.dumps({
                            "type": "deployment_success",
                            "admin_data": {
                                "msg": f"Deployed machine to {env_id}",
                            },
                        })
                    )
                else:
                    print(f"skipping invalid env id: {env_id}")
                    await self.send(
                        text_data=json.dumps({
                            "type": "deployment_error",
                            "admin_data": {
                                "msg": f"skipping invalid env id: {env_id}",
                            },
                        }))
            except Exception as e:
                print(f"Err deploymnt: {e}")
                await self.send(
                    text_data=json.dumps({
                        "type": "deployment_error",
                        "admin_data": {
                            "msg": f"Failed to deploy machine to {env_id}: {str(e)}",
                        },
                    })
                )

    async def ai_log_sum_process(self, data):
        nid = data.get("nid")

        self.logs_explainer = LogAIExplain(
            self.db_manager,
            self.user_id,
        )

        response = self.logs_explainer.analyze_logs(
            nid
        )
        await self.send(text_data=json.dumps({
            "type": "ai_log_sum",
            "message": "success",
            "admin_data": response
        }))


    async def check_ready(self, env_ids:list[str]):
        print("Start ready Thread")

        def _connect():
            """
            Wait till all clusters are build up
            """
            ready_envs:list = []
            try:
                for env_id in env_ids:
                    print("_connect", env_id)
                    data = self.db_manager.get_data(
                        path=f"users/{self.user_id}/env/{env_id}/global_states/",
                    )

                    if "global_states" in data:
                        ready: bool = data["global_states"]["ready"]
                        if ready is True:
                            self.ready_sessions.append(env_id)
                            ready_envs.append(env_id)
                    time.sleep(2)
                    print(f"{len(ready_envs)}/{len(env_ids)}")
                print("Finished Ready check")
                if len(ready_envs) == len(env_ids):
                    return True
                return False
            except Exception as e:
                print(f"Error chck for global state: {e}")
            return False
        if self.testing is True:
            # await readyness
            connected: bool = _connect()


        #
        # FROM HERE FRONTEND HAS A LISTENER
        #

    async def send_env_ids(self):
        print("Send env ids to frontend")
        await self.send(
            text_data=json.dumps(
                {
                    "type": "env_ids",
                    "status": "successful",
                    "admin_data": self.created_envs,
                }
            )
        )


    async def command_handler(
            self,
            data:dict,
    ):
        """
        Deploy a docker in created vm and executes
        """
        classification = self.chat_classifier._classify_input(
            user_input=data.get("text")
        )

        print("classification recieved:", classification)

        if classification in self.chat_classifier.use_cases:
            result = self.chat_classifier.use_cases[classification]
            await self.send(
                text_data=json.dumps({
                    "type": "classification_success",
                    "status": "success",
                    "msg": result,
                })
            )
        else:
            await self.error_response()


    async def log_request_handler(
            self,
            data
    ):
        nid = data.get("nid")

        log_paths = self.get_log_paths(nid)
        out_entries = self.db_manager.get_latest_entries(
            path=log_paths["out"]
        )
        err_entries = self.db_manager.get_latest_entries(
            path=log_paths["err"]
        )
        print(f"Logs for {nid} extracted")
        await self.send(
            text_data=json.dumps(
                {
                    "err": err_entries,
                    "out": out_entries,
                    # todo create listener frontend
                    "path": log_paths,
                }
            )
        )





    async def auth_manager(self,data):
        self.env_id = data.get("env_id")[0]
        print("self.env_id", self.env_id)

        self.auth_data = {
            "type": "auth",
            "admin_data": {
                "session_id": self.session_id,
                "key": self.env_id,
            }
        }

        
    async def set_cluster_vm_ip(
            self,
            env_id
    ) -> str:
        if self.testing is True:
            self.trgt_vm_ip = get_local_ip()
        else:
            #self.trgt_vm_ip = get_vm_public_ip_address(env_id)
            pass




    async def set_received_cluster_creds(self, ws):
        print("Connection to cluster established")
        self.cluster_ws = ws


    def start_bq_thread(self):

        def rcv_data(loop, update_def):
            time.sleep(30)
            data = None

            payload = dict(
                dataset_id=os.environ.get("QDS_ID"),
                table_ids=[],
                target_id=self.session_id
            )

            while data is None:
                try:
                    response = requests.post(
                        self.data_request_endpoint,
                        data=payload
                    )
                    if response.ok:
                        res_data = response.json()

                        print("Data successful collected")
                        loop.call_soon_threadsafe(
                            asyncio.create_task,  # Erstellt eine Task im Event Loop
                            update_def(res_data)
                        )
                except Exception as e:
                    print(f"Error wile reuqest bq admin_data: {e}")
                    time.sleep(5)

        def handle_data(data):
            from asgiref.sync import async_to_sync
            async_to_sync(self.handle_data_response)(data)

        self.data_thread = threading.Thread(
            target=rcv_data,
            args=(self.data_thread_loop, handle_data),
            name=f"DataThread-{self.user_id}",
            daemon=True,  # Optional: Der Thread wird beendet, wenn das Hauptprogramm endet
        )
        self.data_thread.start()


    async def handle_data_response(self, data):
        await self.send(text_data=json.dumps({
            "type": "dataset",
            "message": "success",
            "admin_data": data
        }))
        # end thread after 1s
        self.data_thread.join(1)


    def get_log_paths(self, nid):
        return dict(
            err=f"{self.database}/logs/{nid}/err/",
            out=f"{self.database}/logs/{nid}/out/",
        )


    async def send_creds_frontend(self, listener_paths):
        await self.send(text_data=json.dumps({
            "type": "creds",
            "message": "success",
            "admin_data": {
                "creds": self.db_manager.fb_auth_payload,
                "db_path": os.environ.get("FIREBASE_RTDB"),
                "listener_paths": listener_paths
            },
        }))


    def get_cfg_schema(self) -> dict:
        # returns {pxid: sid: value, phase:[]}
        cfg = {}
        for pixel_id, attrs in self.g.G.nodes(data=True):
            if attrs.get("type") == "PIXEL":
                cfg[pixel_id] = {}
                all_fermion_subs:dict = self.g.get_neighbor_list(
                    node=pixel_id,
                    target_type=[],
                )
                for sid, sattrs in all_fermion_subs.items():
                    energy = sattrs.get("energy")
                    # Provide default value
                    cfg[pixel_id][sid] = {
                        "max_value": energy,
                        "phase": []
                    }
        return cfg

    async def create_frontend_env_content(self):
        nodes = []
        id_map = set()

        for nid, attrs in self.g.G.nodes(data=True):
            if attrs.get("type").lower() not in ["users", "user"]:
                nodes.append(
                    {
                        "id": nid,
                        "pos": attrs.get("pos"),
                        "meta": attrs.get("metadata"),
                    }
                )
                id_map.add(nid)

        print("Nodes extracted", len(nodes))

        edges = [
            {
                "src": src,
                "trgt": trgt,
            }
            for src, trgt, attrs in
            self.g.G.edges(data=True)
            if attrs.get("src_layer").lower() not in ["env", "user", "users"]
            and attrs.get("trgt_layer").lower() not in ["env", "user", "users"]
        ]

        print("Edges extracted", len(edges))

        # EXTRACT PATHS
        all_paths = self.db_manager._get_db_paths_from_G(
            G=self.g.G,
            db_base=self.database,
        )

        empty_nid_struct = {
            nid: {}
            for nid in id_map
        }

        env_content = {
            "type": "init_graph_data",  # todo re-set type front
            "message": "success",
            "admin_data": {
                "edges": edges,
                "nodes": nodes,
                "meta": empty_nid_struct,
                "logs": empty_nid_struct,
            },
        }
        return env_content, all_paths




    async def demo_workflow(self):
        self.sim.env = self.g.G.nodes[TEST_ENV_ID]
        self.sim.run_sim(self.g)
        await self.file_response(
            {"admin_data": self.sim.updator.datastore}
        )
        return

    async def handle_cluster_command(self, c_data):
        if getattr(self, "cluster_auth_data", None) is None:
            data = {
                "type": "auth",
                "admin_data": {
                    "session_id": self.session_id,
                    "key": self.env_id,
                }
            }

            res_data = await self.utils.apost(
                url=self.trgt_vm_domain,
                data=data
            )

        res_data = await self.utils.apost(
            url=self.trgt_vm_domain,
            data=c_data
        )

        print(f"res recvd: {res_data}")

        if "response_key" in res_data:
            print("Auth response received")
            """
            response_key=self.local_key,
            session_id=self.session_id,
            key=key,
            actor_info=self.get_actor_info()
            """
            setattr(self, "cluster_auth_data", res_data)
        elif "type" in res_data and res_data["type"] == "status_success_distribution":
            print(f"response of command distribution received: {res_data}")
            await self.send(text_data=json.dumps({
                "type": "distribution_complete",
                "status": "success",
            }))




    async def file_response(self, content):
        await self.send(
            text_data=json.dumps({
            "type": "data_response",
            "admin_data": content
        }))



    async def disconnect(self, close_code):
        """Called when the websocket is disconnected."""
        # Send message to env node to close
        print("disconnect ws")
        if self.env_node is not None:
            print(f"WebSocket disconnected with code: {close_code}")

    async def _validate_env_state(self, state):
        msg = state["msg"]
        if msg == "unable_fetch_data":
            # close connection
            await self.send(text_data=json.dumps({
                "type": "unable_fetch_data",
                "message": "failed",
            }))




    async def handle_data_changes(self, data):
        # admin_data => {'type': None, 'path': '/', 'admin_data': {'F_mu_
        print("handle_data_changes")
        # todo make a class for it
        all_subs = self.qf_utils.get_all_subs_list(just_id=True)

        attrs = data["admin_data"]
        #print("changed attrs", attrs)
        nid = attrs["id"]

        if attrs is not None:
            if "status" in attrs:  # metadata
                status = data["status"]  # dict: info, state
                state = status["state"]
                for state_type, state_ids in self.worker_states.items():
                    if nid in state_ids and state == state_type:
                        return

                info = status["info"]

                if state not in self.worker_states:
                    self.worker_states[state] = []

                self.worker_states[state].append(nid)
                await self.send(text_data=json.dumps({
                    "type": "metadata_update",
                    "admin_data": {
                        "id": nid,
                        "admin_data": data,
                    }
                }))


            elif "src" in attrs and "trgt" in attrs:
                src = attrs.get("src")
                trgt = attrs.get("trgt")

                eattrs = self.g.G.edges[src, trgt]

                changes = self.check_changes(
                    old=eattrs, new=attrs
                )

                if len(list(changes.keys())):
                    # edge change
                    await self.send(text_data=json.dumps({
                        "type": "edge_data",
                        "admin_data": {
                            "admin_data": data,
                        }
                    }))
                    self.g.G.edges[src, trgt].update(attrs)

            elif "type" in attrs: # node update
                nattrs = self.g.G.nodes[nid]

                changes = self.check_changes(
                    old=nattrs, new=attrs
                )

                if len(list(changes.keys())):
                    for nid in list(self.g.id_map):
                        if nid in attrs["id"]:
                            # todo attr change -> filter edges and create weight
                            # todo filter just necessary key fields (meta field value etc)
                            await self.send(text_data=json.dumps({
                                "type": "node_data",
                                "admin_data": {
                                    "id": nid,
                                    "admin_data": data,
                                }
                            }))
                            break

        if len(self.worker_states["error"]) > 0:
            self.db_manager.upsert_data(
                path=f"{self.database}/global_states/error_nodes/",
                data={nid: info}
            )
        if len(self.worker_states["inactive"]) > 0:
            pass  # todo
        if len(self.worker_states["active"]) == len(all_subs):
            # db global upsert
            self.sim_ready = True
            self.db_manager.upsert_data(
                path=f"{self.database}/global_states/",
                data={"state": "run"}  # ech node listen to it
            )

        print(f"Metadata changes for {nid} sent")


    async def error_response(self):
        print("Classification was not valid")
        await self.send(text_data=json.dumps({
            "type": "classification_error",
            "status": "success",
            "msg": "Invalid Command registered",
        }))

    def check_changes(self, old, new):
        """
        Compare init state
        :param old:
        :param new:
        :return:
        """
        changes = {}
        for k, v in new.items():
            if k in old:
                if new[k] != old[k]:
                    changes[k] = new[k]
        return changes



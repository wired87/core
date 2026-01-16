import importlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx

from core.env_manager.env_lib import handle_get_env, handle_del_env, handle_set_env, handle_get_envs_user, EnvManager, handle_link_env_module, handle_rm_link_env_module, handle_download_model, handle_retrieve_logs_env, handle_get_env_data
from core.injection_manager.injection import (
    handle_set_inj, handle_del_inj, handle_get_inj_user, handle_get_inj_list,
    handle_link_inj_env, handle_rm_link_inj_env, handle_list_link_inj_env, handle_get_injection,
    handle_link_session_injection, handle_rm_link_session_injection, handle_get_sessions_injections
)

from core.method_manager.case import RELAY_METHOD
from core.module_manager.ws_modules_manager import (
    handle_del_module,
    handle_link_session_module,
    handle_set_module,
    handle_get_module,
    handle_list_users_modules,
)
from core.fields_manager.fields_lib import handle_del_field, handle_set_field, \
    handle_link_module_field, handle_list_modules_fields, handle_list_users_fields, \
    handle_rm_link_module_field, handle_get_modules_fields, handle_get_sessions_fields, \
    fields_manager

from core.param_manager.params_lib import (
    handle_get_users_params, handle_set_param, handle_del_param,
    handle_link_field_param, handle_rm_link_field_param, handle_get_fields_params,
)

from core.guard import Guard


import inspect
import json
import dotenv

from channels.generic.websocket import AsyncWebsocketConsumer

import asyncio
from urllib.parse import parse_qs


from core.module_manager.ws_modules_manager.modules_lib import handle_rm_link_session_module
from core.sm_manager.sm_manager import handle_enable_sm
from qf_utils.qf_utils import QFUtils

from _chat.log_sum import LogAIExplain

from workflows.data_distirbutor import DataDistributor
from workflows.deploy_sim import DeploymentHandler
from workflows.node_cfg_manager import NodeCfgManager
from utils.deserialize import deserialize

from _chat.main import AIChatClassifier

from utils.dj_websocket.handler import ConnectionManager


from utils.graph.local_graph_utils import GUtils
from utils.id_gen import generate_id
from utils.utils import Utils


from core.user_manager import UserManager
from core.session_manager.session import SessionManager
from core.session_manager.session import (
    handle_get_sessions_modules, handle_get_sessions_envs,
    handle_link_env_session, handle_rm_link_env_session,
    handle_list_user_sessions
)

from dataclasses import dataclass
from typing import Callable, Any, Awaitable, Dict


@dataclass
class RelayCase:
    case: str
    description: str
    callable: Callable[[Any], Awaitable[Any]]
    required_data_structure: Dict = None
    output_data_structure: Dict = None

dotenv.load_dotenv()

RELAY_CASES_CONFIG = [
    # ENV HANDLING
    {"case": "GET_ENV", "desc": "", "func": handle_get_env, "req_struct": {"auth": {"env_id": "str"}}, "out_struct": {"type": "GET_ENV", "data": {"env": "dict"}}},
    {"case": "GET_USERS_ENVS", "desc": "", "func": handle_get_envs_user, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": "list"}}},
    {"case": "DEL_ENV", "desc": "", "func": handle_del_env, "req_struct": {"auth": {"env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": "list"}}},
    {"case": "SET_ENV", "desc": "", "func": handle_set_env, "req_struct": {"data": {"env_item": "dict"}, "auth": {"user_id": "str"}}, "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": "list"}}},
    {"case": "DOWNLOAD_MODEL", "desc": "Download Model", "func": handle_download_model, "req_struct": {"auth": {"env_id": "string", "user_id": "string"}}, "out_struct": {"type": "DOWNLOAD_MODEL", "data": "dict"}},
    {"case": "RETRIEVE_LOGS_ENV", "desc": "Retrieve Logs Env", "func": handle_retrieve_logs_env, "req_struct": {"auth": {"env_id": "string", "user_id": "string"}}, "out_struct": {"type": "RETRIEVE_LOGS_ENV", "data": "list"}},
    {"case": "GET_ENV_DATA", "desc": "Get Env Data", "func": handle_get_env_data, "req_struct": {"auth": {"env_id": "string", "user_id": "string"}}, "out_struct": {"type": "GET_ENV_DATA", "data": "list"}},
    
    # FIELD
    {"case": "DEL_FIELD", "desc": "Delete Field", "func": handle_del_field, "req_struct": {"auth": {"field_id": "str", "user_id": "str"}}, "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": "list"}}},
    {"case": "LIST_USERS_FIELDS", "desc": "List Users Fields", "func": handle_list_users_fields, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": "list"}}},
    {"case": "LIST_MODULES_FIELDS", "desc": "List Modules Fields", "func": handle_list_modules_fields, "req_struct": {"auth": {"module_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": "list"}}},
    {"case": "SET_FIELD", "desc": "Set Field", "func": handle_set_field, "req_struct": {"data": {"field": "dict"}, "auth": {"field_id": "str", "user_id": "str"}}, "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": "list"}}},
    {"case": "LINK_MODULE_FIELD", "desc": "Link Module Field", "func": handle_link_module_field, "req_struct": {"auth": {"user_id": "str", "module_id": "str", "field_id": "str"}}, "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": "list"}}},
    {"case": "RM_LINK_MODULE_FIELD", "desc": "Remove Link Module Field", "func": handle_rm_link_module_field, "req_struct": {"auth": {"user_id": "str", "module_id": "str", "field_id": "str"}}, "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": "list"}}},
    {"case": "GET_MODULES_FIELDS", "desc": "Get Modules Fields", "func": handle_get_modules_fields, "req_struct": {"auth": {"user_id": "str", "module_id": "str"}}, "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": "list"}}},
    {"case": "SESSIONS_FIELDS", "desc": "Get Sessions Fields", "func": handle_get_sessions_fields, "req_struct": {"auth": {"user_id": "str", "session_id": "str"}}, "out_struct": {"type": "SESSIONS_FIELDS", "data": {"fields": "list"}}},

    # INJECTION - Energy Designer
    {"case": "SET_INJ", "desc": "Set/upsert injection", "func": handle_set_inj, "req_struct": {"data": "dict", "auth": {"user_id": "str"}}, "out_struct": {"type": "GET_INJ_USER", "data": dict[str, list]}},
    {"case": "DEL_INJ", "desc": "Delete injection", "func": handle_del_inj, "req_struct": {"auth": {"injection_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_USER", "data": dict[str, list]}},
    {"case": "GET_INJ_USER", "desc": "Get user injections", "func": handle_get_inj_user, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "GET_INJ_USER", "data": dict[str, list]}},
    {"case": "GET_INJ_LIST", "desc": "Get injection list", "func": handle_get_inj_list, "req_struct": {"data": {"inj_ids": "list"}, "auth": {"user_id": "str"}}, "out_struct": {"type": "GET_INJ_LIST", "data": "list"}},
    
    # INJECTION - Env Linking
    {"case": "LINK_INJ_ENV", "desc": "Link injection to env", "func": handle_link_inj_env, "req_struct": {"auth": {"injection_id": "str", "env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_ENV", "data": "list"}},
    {"case": "RM_LINK_INJ_ENV", "desc": "Remove link injection to env", "func": handle_rm_link_inj_env, "req_struct": {"auth": {"injection_id": "str", "env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_ENV", "data": "list"}},
    {"case": "LIST_LINK_INJ_ENV", "desc": "List env linked injections", "func": handle_list_link_inj_env, "req_struct": {"auth": {"env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_ENV", "data": "list"}},
    {"case": "GET_INJ_ENV", "desc": "Get env injections", "func": handle_list_link_inj_env, "req_struct": {"auth": {"env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_ENV", "data": "list"}},
    {"case": "GET_INJECTION", "desc": "Get single injection", "func": handle_get_injection, "req_struct": {"auth": {"injection_id": "str"}}, "out_struct": {"type": "GET_INJECTION", "data": "dict"}},

    # INJECTIONS (Session)
    {"case": "LINK_SESSION_INJECTION", "desc": "Link Session Injection", "func": handle_link_session_injection, "req_struct": {"auth": {"session_id": "str", "injection_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_INJECTIONS", "data": {"injections": "list"}}},
    {"case": "RM_LINK_SESSION_INJECTION", "desc": "Remove Link Session Injection", "func": handle_rm_link_session_injection, "req_struct": {"auth": {"session_id": "str", "injection_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_INJECTIONS", "data": {"injections": "list"}}},
    {"case": "GET_SESSIONS_INJECTIONS", "desc": "Get Session Injections", "func": handle_get_sessions_injections, "req_struct": {"auth": {"session_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_INJECTIONS", "data": {"injections": "list"}}},

    {"case": "REQUEST_INJ_SCREEN", "desc": "Requesting admin_data relavant for inj setup", "func": None, "func_name": "request_inj_process_start", "req_struct": {"data": {"env_id": "str"}}, "out_struct": {"type": "INJ_PATTERN_STRUCT", "admin_data": "dict", "env_id": "str"}},
    {"case": "SET_INJ_PATTERN", "desc": "Set ncfg injection pattern", "func": None, "func_name": "set_env_inj_pattern", "req_struct": {"data": "dict"}, "out_struct": None},
    {"case": "GET_INJ", "desc": "Retrieve inj cfg list", "func": None, "func_name": "set_env_inj_pattern", "req_struct": {"data": "dict"}, "out_struct": None},

    # SESSION
    {"case": "LINK_ENV_SESSION", "desc": "Link Env Session", "func": handle_link_env_session, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "env_id": "str"}}, "out_struct": {"type": "LINK_ENV_SESSION", "data": {"sessions": "dict"}}},
    {"case": "RM_LINK_ENV_SESSION", "desc": "Remove Link Env Session", "func": handle_rm_link_env_session, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "env_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_ENVS", "data": {"envs": "list"}}},
    {"case": "GET_SESSIONS_ENVS", "desc": "Get Session Envs", "func": handle_get_sessions_envs, "req_struct": {"auth": {"user_id": "str", "session_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_ENVS", "data": {"envs": "list"}}},
    {"case": "LIST_USERS_SESSIONS", "desc": "List user sessions", "func": handle_list_user_sessions, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "LIST_USERS_SESSIONS", "data": {"sessions": "list"}}},
    
    # MODULES
    {"case": "DEL_MODULE", "desc": "Delete Module", "func": handle_del_module, "req_struct": {"auth": {"module_id": "str", "user_id": "str"}}, "out_struct": {"type": "LIST_USERS_MODULES", "data": {"modules": "list"}}},
    {"case": "LINK_SESSION_MODULE", "desc": "Link Session Module", "func": handle_link_session_module, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "module_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_MODULES", "data": {"modules": "list"}}},
    {"case": "RM_LINK_SESSION_MODULE", "desc": "Remove Link Session Module", "func": handle_rm_link_session_module, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "module_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_MODULES", "data": {"modules": "list"}}},
    {"case": "LINK_ENV_MODULE", "desc": "Link Env Module", "func": handle_link_env_module, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "env_id": "str", "module_id": "str"}}, "out_struct": {"type": "LINK_ENV_MODULE", "data": {"sessions": "dict"}}},
    {"case": "RM_LINK_ENV_MODULE", "desc": "Remove Link Env Module", "func": handle_rm_link_env_module, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "env_id": "str", "module_id": "str"}}, "out_struct": {"type": "LINK_ENV_MODULE", "data": {"sessions": "dict"}}},
    {"case": "SET_MODULE", "desc": "Set Module", "func": handle_set_module, "req_struct": {"data": {"id": "str", "files": "list"}, "auth": {"user_id": "str"}}, "out_struct": {"type": "LIST_USERS_MODULES", "data": {"modules": "list"}}},
    {"case": "ENABLE_SM", "desc": "Enable Standard Model", "func": handle_enable_sm, "req_struct": {"auth": {"env_id": "str", "session_id": "str", "user_id": "str"}}, "out_struct": {"type": "ENABLE_SM", "data": {"sessions": "dict"}}},
    {"case": "GET_MODULE", "desc": "Get Module", "func": handle_get_module, "req_struct": {"auth": {"module_id": "str"}}, "out_struct": {"type": "GET_MODULE", "data": "dict"}},
    {"case": "GET_SESSIONS_MODULES", "desc": "Get Session Modules", "func": handle_get_sessions_modules, "req_struct": {"auth": {"user_id": "str", "session_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_MODULES", "data": {"modules": "list"}, "auth": {"session_id": "str"}}},
    {"case": "LIST_USERS_MODULES", "desc": "List User Modules", "func": handle_list_users_modules, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "LIST_USERS_MODULES", "data": {"modules": "list"}}},
    {"case": "CONVERT_MODULE", "desc": "Convert Module", "func": None, "func_name": "_handle_convert_module", "req_struct": {"auth": {"module_id": "str"}, "data": {"files": "dict"}}, "out_struct": {"type": "CONVERT_MODULE", "data": "dict"}},
    # todo apply last paramet entire grid to eq def  - in jax_test
    
    # PARAMS
    {"case": "LIST_USERS_PARAMS", "desc": "Get Users Params", "func": handle_get_users_params, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "LIST_USERS_PARAMS", "data": {"params": "list"}}},
    {"case": "SET_PARAM", "desc": "Set Param", "func": handle_set_param, "req_struct": {"auth": {"user_id": "str"}, "data": {"param": "dict"}}, "out_struct": {"type": "LIST_USERS_PARAMS", "data": {"params": "list"}}},
    {"case": "DEL_PARAM", "desc": "Delete Param", "func": handle_del_param, "req_struct": {"auth": {"user_id": "str", "param_id": "str"}}, "out_struct": {"type": "LIST_USERS_PARAMS", "data": {"params": "list"}}},
    {"case": "LINK_FIELD_PARAM", "desc": "Link Field Param", "func": handle_link_field_param, "req_struct": {"auth": {"user_id": "str"}, "data": {"links": "list"}}, "out_struct": {"type": "GET_FIELDS_PARAMS", "data": {"params": "list"}}},
    {"case": "RM_LINK_FIELD_PARAM", "desc": "Rm Link Field Param", "func": handle_rm_link_field_param, "req_struct": {"auth": {"user_id": "str", "field_id": "str", "param_id": "str"}}, "out_struct": {"type": "GET_FIELDS_PARAMS", "data": {"params": "list"}}},
    {"case": "GET_FIELDS_PARAMS", "desc": "Get Fields Params", "func": handle_get_fields_params, "req_struct": {"auth": {"user_id": "str", "field_id": "str"}}, "out_struct": {"type": "GET_FIELDS_PARAMS", "data": {"params": "list"}}},

    #
    *RELAY_METHOD,
]

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
        self.env_manager = EnvManager()

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
        self.user_manager = UserManager()


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
        self.session_id=None
        self.loop = asyncio.new_event_loop()
        self.data_thread_loop = asyncio.new_event_loop()
        self.user_tables_authenticated = False
        self.cluster_auth_data = None



        self.con_type="http" if os.name == "nt" else "https"
        self.cluster_domain = "127.0.0.1:8001" if os.name == "nt" else "clusterexpress.com"
        self.cluster_url = f"{self.con_type}://{self.cluster_domain}/"
        print(f"Cluster domain set: {self.cluster_url}")

        if self.testing is True:
            self.cluster_root = "http://127.0.0.1:8001"

        else:
            self.cluster_root = "cluster.botworld.cloud"

        self.auth_data = None

        #self.firestore = FirestoreMgr()

        self.relay_cases: list[RelayCase] = RELAY_CASES_CONFIG
        #self._register_cases()


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

        print("self.user_id", self.user_id)

        if not self.user_id:
            print(f"Connection attempt declined")
            await self.close()
            return

        print(f"Connection attempt verified {self.user_id}")

        self.g = GUtils(
            nx_only=False,
            G=nx.Graph(),
            g_from_path=None,
        )

        self.qfu = QFUtils(
            g=self.g
        )

        self.data_distributor = DataDistributor(
            parent=self,
            testing=self.testing,
            user_id=self.user_id,
        )

        self.node_cfg_manager = NodeCfgManager(
            self.user_id,
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

        # Initialize User Management Workflow
        if not self.user_tables_authenticated:
            try:
                user_email = None
                workflow_results = self.user_manager.initialize_qbrain_workflow(
                    uid=self.user_id,
                    email=user_email,
                )
                print(f"User management workflow completed: {workflow_results}")
            except Exception as e:
                print(f"Warning: User management workflow error: {e}")
            self.user_tables_authenticated=True
        
        if not self.session_id:
            # Initialize Session Management and Create Session
            try:
                self.session_manager = SessionManager()
                # Create session after successful user authentication
                session_id = self.session_manager.get_or_create_active_session(self.user_id)
                if session_id:
                    self.session_id = session_id
                    print(f"Session created successfully: {session_id}")
                else:
                    raise Exception(f"Session creation failed for user {self.user_id}")
            except Exception as e:
                print(f"Warning: SESSION MGR ERR: {e}")
                # Generate fallback session_id to ensure functional stability
                self.session_id = generate_id()
                print(f"USING FALLABACK SESSION ID: {self.session_id}")
        await self.send_session()
        print(f"REQUEST FOR USER {self.user_id} ACCEPTED")


    async def send_session(self):
        print("send_session")
        return_data = {
            "type": "SET_SID",
            "auth": {
                "sid": self.session_id,
            },
            "data": {},
        }
        return_data=json.dumps(return_data)
        await self.send(text_data=return_data)
        print("send_session... done")



    def _collect_relay_cases_dynamically(self):
        """
        Scans the project for python files containing variables starting with 'RELAY_',
        loads them, and if they are a list of dicts, extends RELAY_CASES_CONFIG.
        """
        print("Starting dynamic RELAY_ case discovery...")
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Exclude common non-project dirs to save time/errors
        exclude_dirs = {'.venv', 'venv', '.git', '__pycache__', '.idea', 'node_modules', 'build', 'dist'}
        
        for root, dirs, files in os.walk(project_root):
            # Modify dirs in-place to exclude unwanted directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith(".py") and file != os.path.basename(__file__): # Skip self to avoid circular/re-import issues if not careful
                    file_path = os.path.join(root, file)
                    
                    try:
                        # We use importlib to load the module dynamically
                        # Calculate relative path for module import dotted notation
                        rel_path = os.path.relpath(file_path, project_root)
                        module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                        
                        # Guard against potential import errors or side effects
                        # In a production system, static analysis (AST) might be safer than importing,
                        # but importing ensures we get the actual evaluated object.
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module # Optional: cache it
                            spec.loader.exec_module(module)
                            
                            for var_name, value in vars(module).items():
                                if var_name.startswith("RELAY_") and isinstance(value, list):
                                    # Basic validation: check if list contains dicts with 'case' key
                                    if value and isinstance(value[0], dict) and "case" in value[0]:
                                        print(f"Found {var_name} in {module_name} with {len(value)} cases.")
                                        for case_def in value:
                                            # Avoid duplicates if case already exists?
                                            # For now, append/overwrite logic could be added. 
                                            # Here we just extend the config list.
                                            
                                            # Check if already registered to avoid dupes from imports
                                            # (Actually RELAY_CASES_CONFIG is global in this file scope, but we want to populate self.relay_cases)
                                            func = case_def.get("func")
                                            # Handle case where func is a lambda or partial which might not pickle well, but here we just run it.
                                            
                                            self.relay_cases.append(RelayCase(
                                                case=case_def["case"], 
                                                description=case_def.get("desc", ""), 
                                                callable=func, 
                                                required_data_structure=case_def.get("req_struct", {}), 
                                                output_data_structure=case_def.get("out_struct", {})
                                            ))
                    except Exception as e:
                        # print(f"Skipping {file_path} due to import error: {e}")
                        pass
        print(f"Dynamic discovery finished. Total cases: {len(self.relay_cases)}")

    def _register_cases(self):
        # First, register the hardcoded/legacy config present in this file
        for case_info in RELAY_CASES_CONFIG:
            func = case_info.get("func")
            self.relay_cases.append(RelayCase(
                case=case_info["case"], 
                description=case_info["desc"], 
                callable=func, 
                required_data_structure=case_info.get("req_struct", {}), 
                output_data_structure=case_info.get("out_struct", {})
            ))
            
        # Then, run dynamic discovery
        #self._collect_relay_cases_dynamically()





    async def receive(
            self,
            text_data=None,
            bytes_data=None
    ):
        print(f"start receive: {text_data}")
        try:
            payload = deserialize(text_data)
            data_type = payload.get("type")  # assuming 'type' field for command
            
            # Inject session_id into auth if available
            if self.session_id:
                if "auth" not in payload:
                    payload["auth"] = {}
                # Only inject if not already present (frontend might override)
                if "session_id" not in payload["auth"]:
                    payload["auth"]["session_id"] = self.session_id
                
            print(f"Received message from frontend: {payload}")

            # Dynamic Dispatch
            handled = False

            # {"case": "START_SIM", "desc": "Start simulation", "func": None, "func_name": "_handle_start_sim_process", "req_struct": {"data": "dict"}, "out_struct": {"type": "START_SIM", "status": {"state": "str", "code": "int", "msg": "str"}, "data": "dict"}},
            if data_type == "START_SIM":
                await self._handle_start_sim_process(payload)
                return

            # Iterate over a copy to ensure thread safety if cases are added at runtime
            for relay_case in list(self.relay_cases):
                if relay_case.get("case") == data_type:
                    # Sig check to support both legacy (1 arg) and new (2 args) handlers
                    if not relay_case.get("func"):
                         if relay_case.get("func_name"):
                             handler = getattr(self, relay_case["func_name"], None)
                         else:
                             print(f"Skipping case {relay_case.get('case')} with no handler.")
                             continue
                    else:
                        handler = relay_case["func"]

                    # Sig check to support both legacy (1 arg) and new (2 args) handlers
                    sig = inspect.signature(handler)
                    try:
                        if len(sig.parameters) >= 2:
                            # if handler.__module__.endswith("env_lib"): # Example logic check
                            res = await handler(payload) if inspect.iscoroutinefunction(handler) else handler(payload)
                        else:
                            res = await handler(payload) if inspect.iscoroutinefunction(handler) else handler(payload)
                    except TypeError as e:
                         # Fallback or re-raise
                         print(f"Error calling handler {relay_case.get('case')}: {e}")
                         raise

                    print("send", res["type"])
                    return_data = json.dumps(res, default=str)

                    await self.send(text_data=return_data)
                    handled = True
                    break
            
            if not handled:
                print(f"Unknown command type received: {data_type}")

        except Exception as e:
            print(f">>Error processing received message: {e}")
            import traceback
            traceback.print_exc()


    async def _handle_start_sim_process(self, payload: dict):
        """
        Handles the START_SIM case.
        Delegates to Guard.sim_start_process to fetch data, build graph, and compile pattern.
        Deactivates session upon success.
        """
        print("Starting Simulation Process with payload:", payload)
        try:
            # 1. Call Guard Process
            self.guard.sim_start_process(payload)
            
            # Update env status to IN_PROGRESS and notify frontend
            env_id = payload.get("data", {}).get("env_id") or payload.get("auth", {}).get("env_id")
            if env_id:
                print(f"Updating status for env {env_id} to IN_PROGRESS")
                try:
                    # Update status
                    self.user_manager.qb.set_item(
                        "envs", 
                        {"status": "IN_PROGRESS"}, 
                        keys={"id": env_id, "user_id": self.user_id}
                    )
                    
                    # Fetch updated entry
                    updated_env = self.user_manager.qb.row_from_id(
                        nid=env_id,
                        table="envs",
                        user_id=self.user_id
                    )
                    
                    if updated_env:
                        # Send to frontend
                        await self.send(text_data=json.dumps({
                            "type": "GET_USERS_ENVS",
                            "data": {"envs": updated_env}
                        }))
                        print(f"Sent IN_PROGRESS status for env {env_id}")
                except Exception as ex:
                    print(f"Error updating env status: {ex}")
            
            # 2. Deactivate Session
            if hasattr(self, 'session_id') and self.session_id:
                print(f"Deactivating session {self.session_id} after successful sim start")
                self.session_manager.deactivate_session(self.session_id)
                self.session_id = self.session_manager.get_or_create_active_session(self.user_id)
            
            # 3. Send Success Response
            response = {
                "type": "START_SIM",
                "status": {
                    "state": "success",
                    "code": 200,
                    "msg": "Simulation started and session completed."
                },
                "data": {} 
            }
            
            await self.send(text_data=json.dumps(response))
            await self.send_session()

            print("Sim start process completed successfully.")

        except Exception as e:
            print(f"Error in start_sim_process: {e}")
            import traceback
            traceback.print_exc()
            
            # Send Error Response
            error_response = {
                "type": "START_SIM",
                "status": {
                    "state": "error",
                    "code": 500,
                    "msg": str(e)
                },
                "data": {}
            }
            await self.send(text_data=json.dumps(error_response))



    async def batch_inject_env(self, payload):
        print("START SIM REQUEST RECEIVED")
        data = payload.get("data", {})
        config = payload.get("config", {})
        if not config and "data" in payload and "config" in payload["data"]:
            config = payload["data"]["config"]
        elif not config and "config" in data:
            config = data["config"]

        for env_id, env_data in config.items():
            try:
                if self.world_creator.env_id_map:

                    self.guard.sim_start_process(
                        env_id,
                        env_data
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



    async def disconnect(self, close_code):
        """Called when the websocket is disconnected."""
        # Send message to env node to close
        print("disconnect ws")
        if self.env_node is not None:
            print(f"WebSocket disconnected with code: {close_code}")




    async def error_response(self):
        print("Classification was not valid")
        await self.send(text_data=json.dumps({
            "type": "classification_error",
            "status": "success",
            "msg": "Invalid Command registered",
        }))




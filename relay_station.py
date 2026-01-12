import os
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx

from core.env_manager.env_lib import handle_get_env, handle_del_env, handle_set_env, handle_get_envs_user, EnvManager, handle_link_env_module, handle_rm_link_env_module, handle_download_model, handle_retrieve_logs_env, handle_get_env_data
from core.injection_manager.injection import (
    handle_set_inj, handle_del_inj, handle_get_inj_user, handle_get_inj_list,
    handle_link_inj_env, handle_rm_link_inj_env, handle_list_link_inj_env, handle_get_injection,
    handle_link_session_injection, handle_rm_link_session_injection, handle_get_sessions_injections
)
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
    params_manager
)

from core.guard import Guard


import inspect
import requests
import json
import dotenv
import zipfile
import io
import pandas as pd
from channels.generic.websocket import AsyncWebsocketConsumer

import asyncio
from urllib.parse import parse_qs


from core.module_manager.ws_modules_manager.modules_lib import handle_rm_link_session_module
from core.module_manager.module_etractor.extractor import RawModuleExtractor
from core.sm_manager.sm_manager import SMManager, handle_enable_sm
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
        self.sm_manager = SMManager()
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

        #self.firestore = FirestoreMgr()

        self.relay_cases: list[RelayCase] = []
        self._register_cases()


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

        try:
            fields_manager.upload_sm_fields(self.user_id)
        except Exception as e:
            print(f"Error uploading SM fields: {e}")

        try:
            params_manager.upload_sm_params(self.user_id)
        except Exception as e:
            print(f"Error uploading SM params: {e}")

        # todo collect more sim admin_data like len, elements, ...
        # todo improve auth
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
                import traceback
                traceback.print_exc()
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
                import traceback
                traceback.print_exc()
                # Generate fallback session_id to ensure functional stability
                self.session_id = generate_id()
                print(f"USING FALLABACK SESSION ID: {self.session_id}")

        await self.send_session()
        print(f"REQUEST FOR USER {self.user_id} ACCEPTED")

    async def send_session(self):
        print("send_session")
        return_data = {
            "type": "SET_SID",
            "auth":{
                "sid": self.session_id,
            },
            "data":{},
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
            if func is None and "func_name" in case_info:
                func = getattr(self, case_info["func_name"])
            
            self.relay_cases.append(RelayCase(
                case=case_info["case"], 
                description=case_info["desc"], 
                callable=func, 
                required_data_structure=case_info.get("req_struct", {}), 
                output_data_structure=case_info.get("out_struct", {})
            ))
            
        # Then, run dynamic discovery
        self._collect_relay_cases_dynamically()





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
                if relay_case.case == data_type:
                    # Sig check to support both legacy (1 arg) and new (2 args) handlers
                    sig = inspect.signature(relay_case.callable)
                    try:
                        if len(sig.parameters) >= 2:
                            if relay_case.callable.__module__.endswith("env_lib"):
                                res = await relay_case.callable(payload) if inspect.iscoroutinefunction(relay_case.callable) else relay_case.callable(payload)
                            else:
                                res = await relay_case.callable(payload) if inspect.iscoroutinefunction(relay_case.callable) else relay_case.callable(payload)
                        else:
                            res = await relay_case.callable(payload) if inspect.iscoroutinefunction(relay_case.callable) else relay_case.callable(payload)
                    except TypeError as e:
                         # Fallback or re-raise
                         print(f"Error calling handler {relay_case.case}: {e}")
                         raise

                    return_data = json.dumps(res, default=str)
                    print("send", return_data)

                    await self.send(text_data=return_data)
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


    async def set_env_inj_pattern(self, data):
        """
        Set injection pattern for environment.
        Handles 'injection_id_struct' by resolving injection IDs to actual data patterns.
        
        Args:
            data: {
                "env_id": str,
                "injection_id_struct": {ntype: {pos_str: inj_id}},
                ...
            }
        """
        print("inj_cfg_process start")
        env_id = data.get("env_id")
        
        # New structure: map mapping position to injection_id
        injection_id_struct = data.get("injection_id_struct") 
        
        resolved_pattern = {}
        
        if injection_id_struct and env_id:
            print(f"Resolving injection IDs for env {env_id}")
            
            # Resolve IDs to patterns
            # Structure: {ntype: {pos: inj_id}}
            for ntype, pos_map in injection_id_struct.items():
                resolved_pattern[ntype] = {}
                
                for pos_key, inj_id in pos_map.items():
                    # Retrieve injection data from BigQuery
                    inj_record = self.injection_manager.get_injection(inj_id)
                    
                    if inj_record and "data" in inj_record:
                        # inj_record["data"] is [[times], [energies]]
                        resolved_pattern[ntype][pos_key] = inj_record["data"]
                    else:
                        print(f"Warning: Injection {inj_id} not found or has no data")
            
            # Apply resolved pattern to guard
            # Note: guard.set_inj_pattern likely expects the pattern data directly
            self.guard.set_inj_pattern(
                resolved_pattern,
                env_id
            )
            print("Injection pattern applied")
    
    # Old _handle methods for injections removed as they are now in injection.py

            
        elif data.get("inj_pattern"):
            # Legacy/Direct pattern support
            self.guard.set_inj_pattern(
                data.get("inj_pattern"),
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
        """
        RCV FILE -> load self.root -> loop all files (check mod eists)
        """
        # HANDLE FILES
        files = data.get("files")
        if files:
            # SAVE RECEIVED FILE
            for f in files:
                with open(os.path.join(self.root, f.name), "w") as content:
                    content.write(f)

            self.guard.handle_mod_stack(
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



    def handle_files(self, files):
        if len(files):
            for f in files:
                f_bytes = f.read()
                name = ask_chat(
                    prompt="Create a unique name for the provided file",
                    base64_string=f_bytes
                )
                print("Uploaded file:", name)


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

    async def _handle_convert_module(self, payload):
        """
        Input: CONVERT_MODULE, auth={module_id, user_id}, data={files={pdf:files}}
        Behavior: Call RawModuleExtractor.process -> return result.
        Output: type=CONVERT_MODULE, data={params:..., code:...}
        """
        auth = payload.get("auth", {})
        data = payload.get("data", {})
        module_id = auth.get("module_id")
        
        if not module_id:
             error_response = {
                "type": "CONVERT_MODULE",
                "status": {"state": "error", "code": 400, "msg": "Missing module_id"},
                "data": {}
            }
             await self.send(text_data=json.dumps(error_response))
             return

        try:
            print(f"Starting module conversion for module {module_id}")
            extractor = RawModuleExtractor(self.user_id, module_id)
            # Run in executor to avoid blocking event loop
            files = data.get("files", {})
            result = await self.loop.run_in_executor(None, extractor.process, files)
            
            response = {
                "type": "CONVERT_MODULE",
                "status": {"state": "success", "code": 200, "msg": "Module converted"},
                "data": result
            }
            await self.send(text_data=json.dumps(response))
            print(f"Module conversion completed for {module_id}")
            
        except Exception as e:
            print(f"Error converting module: {e}")
            import traceback
            traceback.print_exc()
            error_response = {
                "type": "CONVERT_MODULE",
                "status": {"state": "error", "code": 500, "msg": str(e)},
                "data": {}
            }
            await self.send(text_data=json.dumps(error_response))



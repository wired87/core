import importlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx

from core.env_manager.env_lib import handle_get_env, handle_del_env, handle_set_env, handle_get_envs_user, EnvManager, handle_link_env_module, handle_rm_link_env_module, handle_download_model, handle_retrieve_logs_env, handle_get_env_data
from core.file_manager import RELAY_FILE
from core.injection_manager.injection import (
    handle_set_inj, handle_del_inj, handle_get_inj_user, handle_get_inj_list,
    handle_link_inj_env, handle_rm_link_inj_env, handle_list_link_inj_env, handle_get_injection,
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
    handle_rm_link_module_field, handle_get_modules_fields, handle_get_sessions_fields

from core.param_manager.params_lib import (
    handle_get_users_params, handle_set_param, handle_del_param,
    #handle_link_field_param, handle_rm_link_field_param, handle_get_fields_params,
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


from utils.deserialize import deserialize

from chat_manger.main import AIChatClassifier

from utils.dj_websocket.handler import ConnectionManager


from utils.graph.local_graph_utils import GUtils
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

# Debug prefix for relay manager methods (grep-friendly)
_RELAY_DEBUG = "[Relay]"


RELAY_CASES_CONFIG = [
    # ENV
    {"case": "GET_ENV", "desc": "", "func": handle_get_env,
     "req_struct": {"auth": {"env_id": str}},
     "out_struct": {"type": "GET_ENV", "data": {"env": dict}}},

    {"case": "GET_USERS_ENVS", "desc": "", "func": handle_get_envs_user,
     "req_struct": {"auth": {"user_id": str}},
     "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": list}}},

    {"case": "DEL_ENV", "desc": "", "func": handle_del_env,
     "req_struct": {"auth": {"env_id": str, "user_id": str}},
     "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": list}}},

    {"case": "SET_ENV", "desc": "", "func": handle_set_env,
     "req_struct": {"data": {"env": dict}, "auth": {"user_id": str}},
     "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": list}}},

    {"case": "DOWNLOAD_MODEL", "desc": "Download Model", "func": handle_download_model,
     "req_struct": {"auth": {"env_id": str, "user_id": str}},
     "out_struct": {"type": "DOWNLOAD_MODEL", "data": dict}},

    {"case": "RETRIEVE_LOGS_ENV", "desc": "Retrieve Logs Env", "func": handle_retrieve_logs_env,
     "req_struct": {"auth": {"env_id": str, "user_id": str}},
     "out_struct": {"type": "RETRIEVE_LOGS_ENV", "data": list}},

    {"case": "GET_ENV_DATA", "desc": "Get Env Data", "func": handle_get_env_data,
     "req_struct": {"auth": {"env_id": str, "user_id": str}},
     "out_struct": {"type": "GET_ENV_DATA", "data": list}},

    # FIELD
    {"case": "DEL_FIELD", "desc": "Delete Field", "func": handle_del_field,
     "req_struct": {"auth": {"field_id": str, "user_id": str}},
     "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": list}}},

    {"case": "LIST_USERS_FIELDS", "desc": "List Users Fields", "func": handle_list_users_fields,
     "req_struct": {"auth": {"user_id": str}},
     "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": list}}},

    {"case": "LIST_MODULES_FIELDS", "desc": "List Modules Fields", "func": handle_list_modules_fields,
     "req_struct": {"auth": {"module_id": str, "user_id": str}},
     "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": list}}},

    {"case": "SET_FIELD", "desc": "Set Field", "func": handle_set_field,
     "req_struct": {"data": {"field": dict}, "auth": {"user_id": str, "original_id": "str"}},
     "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": list}}},

    {"case": "LINK_MODULE_FIELD", "desc": "Link Module Field", "func": handle_link_module_field,
     "req_struct": {"auth": {"user_id": str, "module_id": str, "field_id": str, "session_id": str, "env_id": str}},
     "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": list}}},

    {"case": "RM_LINK_MODULE_FIELD", "desc": "Remove Link Module Field", "func": handle_rm_link_module_field,
     "req_struct": {"auth": {"user_id": str, "module_id": str, "field_id": str}},
     "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": list}}},

    {"case": "GET_MODULES_FIELDS", "desc": "Get Modules Fields", "func": handle_get_modules_fields,
     "req_struct": {"auth": {"user_id": str, "module_id": str}},
     "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": list}}},

    {"case": "SESSIONS_FIELDS", "desc": "Get Sessions Fields", "func": handle_get_sessions_fields,
     "req_struct": {"auth": {"user_id": str, "session_id": str}},
     "out_struct": {"type": "SESSIONS_FIELDS", "data": {"fields": list}}},

    # INJECTION – Energy Designer
    {"case": "SET_INJ", "desc": "Set/upsert injection", "func": handle_set_inj,
     "req_struct": {"data": dict, "auth": {"user_id": str, "original_id": str}},
     "out_struct": {"type": "GET_INJ_USER", "data": dict[str, list]}},

    {"case": "DEL_INJ", "desc": "Delete injection", "func": handle_del_inj,
     "req_struct": {"auth": {"injection_id": str, "user_id": str}},
     "out_struct": {"type": "GET_INJ_USER", "data": dict[str, list]}},

    {"case": "GET_INJ_USER", "desc": "Get user injections", "func": handle_get_inj_user,
     "req_struct": {"auth": {"user_id": str}},
     "out_struct": {"type": "GET_INJ_USER", "data": dict[str, list]}},

    {"case": "GET_INJ_LIST", "desc": "Get injection list", "func": handle_get_inj_list,
     "req_struct": {"data": {"inj_ids": list}, "auth": {"user_id": str}},
     "out_struct": {"type": "GET_INJ_LIST", "data": list}},

    # INJECTION – Env Linking
    {"case": "LINK_INJ_ENV", "desc": "Link injection to env", "func": handle_link_inj_env,
     "req_struct": {"auth": {"injection_id": str, "env_id": str, "user_id": str}},
     "out_struct": {"type": "GET_INJ_ENV", "data": list}},

    {"case": "RM_LINK_INJ_ENV", "desc": "Remove link injection to env", "func": handle_rm_link_inj_env,
     "req_struct": {"auth": {"injection_id": str, "env_id": str, "user_id": str}},
     "out_struct": {"type": "GET_INJ_ENV", "data": list}},

    {"case": "LIST_LINK_INJ_ENV", "desc": "List env linked injections", "func": handle_list_link_inj_env,
     "req_struct": {"auth": {"env_id": str, "user_id": str}},
     "out_struct": {"type": "GET_INJ_ENV", "data": list}},

    {"case": "GET_INJECTION", "desc": "Get single injection", "func": handle_get_injection,
     "req_struct": {"data": {"id": str, "injection_id": str}, "auth": {}},
     "out_struct": {"type": "GET_INJECTION", "data": dict}},

    # SESSION
    {"case": "LIST_USERS_SESSIONS", "desc": "List user sessions", "func": handle_list_user_sessions,
     "req_struct": {"auth": {"user_id": str}},
     "out_struct": {"type": "LIST_USERS_SESSIONS", "data": {"sessions": list}}},


    # MDULE
    {"case": "DEL_MODULE", "desc": "Delete Module", "func": handle_del_module,
     "req_struct": {"auth": {"module_id": str, "user_id": str}},
     "out_struct": {"type": "LIST_USERS_MODULES", "data": {"modules": list}}},

    {"case": "LINK_SESSION_MODULE", "desc": "Link Session Module", "func": handle_link_session_module,
     "req_struct": {"auth": {"user_id": str, "session_id": str, "module_id": str}},
     "out_struct": {"type": "GET_SESSIONS_MODULES", "data": {"modules": list}}},

    {"case": "RM_LINK_SESSION_MODULE", "desc": "Remove Link Session Module", "func": handle_rm_link_session_module,
     "req_struct": {"auth": {"user_id": str, "session_id": str, "module_id": str}},
     "out_struct": {"type": "GET_SESSIONS_MODULES", "data": {"modules": list}}},

    {"case": "LINK_ENV_MODULE", "desc": "Link Env Module", "func": handle_link_env_module,
     "req_struct": {"auth": {"user_id": str, "session_id": str, "env_id": str, "module_id": str}},
     "out_struct": {"type": "LINK_ENV_MODULE", "data": {"sessions": dict}}},

    {"case": "RM_LINK_ENV_MODULE", "desc": "Remove Link Env Module", "func": handle_rm_link_env_module,
     "req_struct": {"auth": {"user_id": str, "session_id": str, "env_id": str, "module_id": str}},
     "out_struct": {"type": "LINK_ENV_MODULE", "data": {"sessions": dict}}},

    {"case": "SET_MODULE", "desc": "Set Module", "func": handle_set_module,
     "req_struct": {"data": {"id": str, "fields": list, "methods": list, "description": str}, "auth": {"user_id": str, "original_id": str}},
     "out_struct": {"type": "LIST_USERS_MODULES", "data": {"modules": list}}},

    {"case": "ENABLE_SM", "desc": "Enable Standard Model", "func": handle_enable_sm,
     "req_struct": {"auth": {"env_id": str, "session_id": str, "user_id": str}},
     "out_struct": {"type": "ENABLE_SM", "data": {"sessions": dict}}},

    {"case": "GET_MODULE", "desc": "Get Module", "func": handle_get_module,
     "req_struct": {"auth": {"module_id": str}},
     "out_struct": {"type": "GET_MODULE", "data": dict}},

    {"case": "GET_SESSIONS_MODULES", "desc": "Get Session Modules", "func": handle_get_sessions_modules,
     "req_struct": {"auth": {"user_id": str, "session_id": str}},
     "out_struct": {"type": "GET_SESSIONS_MODULES", "data": {"modules": list}, "auth": {"session_id": str}}},

    {"case": "LIST_USERS_MODULES", "desc": "List User Modules", "func": handle_list_users_modules,
     "req_struct": {"auth": {"user_id": str}},
     "out_struct": {"type": "LIST_USERS_MODULES", "data": {"modules": list}}},

    {"case": "CONVERT_MODULE", "desc": "Convert Module", "func": None, "func_name": "_handle_convert_module",
     "req_struct": {"auth": {"module_id": str}, "data": {"files": dict}},
     "out_struct": {"type": "CONVERT_MODULE", "data": dict}},





    # PARAMS
    {"case": "LIST_USERS_PARAMS", "desc": "Get Users Params", "func": handle_get_users_params,
     "req_struct": {"auth": {"user_id": str}},
     "out_struct": {"type": "LIST_USERS_PARAMS", "data": {"params": list}}},

    {"case": "SET_PARAM", "desc": "Set Param", "func": handle_set_param,
     "req_struct": {"auth": {"user_id": str, "original_id": str}, "data": {"param": "dict|list"}},
     "out_struct": {"type": "LIST_USERS_PARAMS", "data": {"params": list}}},

    {"case": "DEL_PARAM", "desc": "Delete Param", "func": handle_del_param,
     "req_struct": {"auth": {"user_id": str, "param_id": str}},
     "out_struct": {"type": "LIST_USERS_PARAMS", "data": {"params": list}}},

    # AIChat handled manually

    #
    *RELAY_METHOD,
    *RELAY_FILE,
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
        self.env_manager = EnvManager()

        self.utils = Utils()

        # save fiel names to apply to envs
        self.file_store:list[str] = []
        self.qc = False
        #self.instance = os.environ["FIREBASE_RTDB"]
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
        self.chat_classifier = AIChatClassifier(case_struct=self.relay_cases)


    async def connect(self):
        try:
            print(f"{_RELAY_DEBUG} connect: connection attempt registered")
            await self.accept()
            print(f"{_RELAY_DEBUG} connect: scope type = {self.scope.get('type')}")

            query_string = self.scope["query_string"].decode()
            query_params = parse_qs(query_string)
            print(f"{_RELAY_DEBUG} connect: query_params = {query_params}")

            user_id = query_params.get("user_id")
            if not user_id:
                print(f"{_RELAY_DEBUG} connect: missing user_id in query; declining")
                await self.close()
                return
            user_id = user_id[0]
            self.user_id = user_id
            print(f"{_RELAY_DEBUG} connect: user_id = {self.user_id}")

            if not self.user_id:
                print(f"{_RELAY_DEBUG} connect: connection declined (no user_id)")
                await self.close()
                return
            print(f"{_RELAY_DEBUG} connect: connection verified for user_id={self.user_id}")

            print(f"{_RELAY_DEBUG} connect: initializing GUtils, QFUtils, DataDistributor, NodeCfgManager, Guard")
            self.g = GUtils(
                nx_only=False,
                G=nx.Graph(),
                g_from_path=None,
            )
            self.qfu = QFUtils(g=self.g)

            self.guard = Guard(
                self.qfu,
                self.g,
                self.user_id,
            )
            print(f"{_RELAY_DEBUG} connect: core components initialized")

            if not self.user_tables_authenticated:
                try:
                    print(f"{_RELAY_DEBUG} connect: running user_manager.initialize_qbrain_workflow")
                    user_email = None
                    workflow_results = self.user_manager.initialize_qbrain_workflow(
                        uid=self.user_id,
                        email=user_email,
                    )
                    print(f"{_RELAY_DEBUG} connect: user workflow completed: {workflow_results}")
                except Exception as e:
                    print(f"{_RELAY_DEBUG} connect: user workflow error (continuing): {e}")
                    import traceback
                    traceback.print_exc()
                self.user_tables_authenticated = True

            if not self.session_id:
                try:
                    print(f"{_RELAY_DEBUG} connect: initializing SessionManager and creating session")
                    self.session_manager = SessionManager()
                    session_id = self.session_manager.get_or_create_active_session(self.user_id)
                    if session_id:
                        self.session_id = session_id
                        print(f"{_RELAY_DEBUG} connect: session created: {session_id}")
                    else:
                        raise Exception(f"Session creation returned None for user {self.user_id}")
                except Exception as e:
                    print(f"{_RELAY_DEBUG} connect: session creation error (using fallback): {e}")
                    self.close()

            print(f"{_RELAY_DEBUG} connect: sending session to client")
            await self.send_session()
            print(f"{_RELAY_DEBUG} connect: request for user {self.user_id} ACCEPTED")
        except Exception as e:
            print(f"{_RELAY_DEBUG} connect: FATAL error: {e}")
            import traceback
            traceback.print_exc()
            try:
                await self.close()
            except Exception as close_err:
                print(f"{_RELAY_DEBUG} connect: error during close: {close_err}")


    async def send_session(self):
        try:
            print(f"{_RELAY_DEBUG} send_session: building SET_SID payload (sid={getattr(self, 'session_id', None)})")
            return_data = {
                "type": "SET_SID",
                "auth": {"sid": self.session_id},
                "data": {},
            }
            return_data = json.dumps(return_data)
            await self.send(text_data=return_data)
            print(f"{_RELAY_DEBUG} send_session: sent SET_SID successfully")
        except Exception as e:
            print(f"{_RELAY_DEBUG} send_session: error: {e}")
            import traceback
            traceback.print_exc()
            raise



    def _collect_relay_cases_dynamically(self):
        """
        Scans the project for python files containing variables starting with 'RELAY_',
        loads them, and if they are a list of dicts, extends RELAY_CASES_CONFIG.
        """
        try:
            print(f"{_RELAY_DEBUG} _collect_relay_cases_dynamically: starting discovery")
            project_root = os.path.dirname(os.path.abspath(__file__))
            exclude_dirs = {'.venv', 'venv', '.git', '__pycache__', '.idea', 'node_modules', 'build', 'dist'}
            initial_count = len(self.relay_cases)

            for root, dirs, files in os.walk(project_root):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for file in files:
                    if not file.endswith(".py") or file == os.path.basename(__file__):
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        rel_path = os.path.relpath(file_path, project_root)
                        module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        if not spec or not spec.loader:
                            continue
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        for var_name, value in vars(module).items():
                            if not var_name.startswith("RELAY_") or not isinstance(value, list):
                                continue
                            if not value or not isinstance(value[0], dict) or "case" not in value[0]:
                                continue
                            print(f"{_RELAY_DEBUG} _collect_relay_cases_dynamically: found {var_name} in {module_name} ({len(value)} cases)")
                            for case_def in value:
                                func = case_def.get("func")
                                self.relay_cases.append(RelayCase(
                                    case=case_def["case"],
                                    description=case_def.get("desc", ""),
                                    callable=func,
                                    required_data_structure=case_def.get("req_struct", {}),
                                    output_data_structure=case_def.get("out_struct", {})
                                ))
                    except Exception as e:
                        print(f"{_RELAY_DEBUG} _collect_relay_cases_dynamically: skip {file_path}: {e}")
            print(f"{_RELAY_DEBUG} _collect_relay_cases_dynamically: finished. total cases: {initial_count} -> {len(self.relay_cases)}")
        except Exception as e:
            print(f"{_RELAY_DEBUG} _collect_relay_cases_dynamically: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _register_cases(self):
        try:
            print(f"{_RELAY_DEBUG} _register_cases: registering {len(RELAY_CASES_CONFIG)} config cases")
            for case_info in RELAY_CASES_CONFIG:
                func = case_info.get("func")
                self.relay_cases.append(RelayCase(
                    case=case_info["case"],
                    description=case_info["desc"],
                    callable=func,
                    required_data_structure=case_info.get("req_struct", {}),
                    output_data_structure=case_info.get("out_struct", {})
                ))
            print(f"{_RELAY_DEBUG} _register_cases: done. total cases: {len(self.relay_cases)}")
        except Exception as e:
            print(f"{_RELAY_DEBUG} _register_cases: error: {e}")
            import traceback
            traceback.print_exc()
            raise





    async def receive(
            self,
            text_data=None,
            bytes_data=None
    ):
        try:
            #print(f"{_RELAY_DEBUG} receive: raw text_data length={len(text_data) if text_data else 0}")
            payload = deserialize(text_data)
            data_type = payload.get("type")
            #print(f"{_RELAY_DEBUG} receive: type={data_type}")

            if self.session_id:
                if "auth" not in payload:
                    payload["auth"] = {}
                if "session_id" not in payload["auth"]:
                    payload["auth"]["session_id"] = self.session_id
            #print(f"{_RELAY_DEBUG} receive: payload prepared (session_id injected if needed)")

            handled = False

            if data_type == "START_SIM":
                print(f"{_RELAY_DEBUG} receive: dispatching to _handle_start_sim_process")
                await self._handle_start_sim_process(payload)
                return

            if data_type == "CHAT":
                data = payload.get("data", {}) or {}

                try:
                    response = self.chat_classifier.chat(
                        self.user_id,
                        user_input=payload["data"]["msg"],
                    )
                    return_data = json.dumps(response, default=str)
                    await self.send(text_data=return_data)
                    print(f"{_RELAY_DEBUG} receive: CHAT response sent (type={response.get('type')})")
                except Exception as chat_err:
                    print(f"{_RELAY_DEBUG} receive: CHAT handler error: {chat_err}")
                    import traceback
                    traceback.print_exc()
                    await self.send(text_data=json.dumps({
                        "type": "CHAT",
                        "status": {"state": "error", "msg": str(chat_err)},
                        "data": {},
                    }, default=str))
                return

            # Relay cases: support both dict (from RELAY_CASES_CONFIG) and RelayCase
            for relay_case in list(self.relay_cases):
                case_name = relay_case.get("case") if isinstance(relay_case, dict) else getattr(relay_case, "case", None)
                if case_name != data_type:
                    continue
                print(f"{_RELAY_DEBUG} receive: matching case {case_name}")
                if not relay_case.get("func") if isinstance(relay_case, dict) else (getattr(relay_case, "callable", None) is None):
                    func_name = relay_case.get("func_name") if isinstance(relay_case, dict) else None
                    if func_name:
                        handler = getattr(self, func_name, None)
                    else:
                        print(f"{_RELAY_DEBUG} receive: skipping case {case_name} (no handler)")
                        continue
                else:
                    handler = relay_case.get("func") if isinstance(relay_case, dict) else getattr(relay_case, "callable", None)
                if handler is None:
                    print(f"{_RELAY_DEBUG} receive: no callable for case {case_name}; skipping")
                    continue
                try:
                    sig = inspect.signature(handler)
                    if len(sig.parameters) >= 2:
                        res = await handler(payload) if inspect.iscoroutinefunction(handler) else handler(payload)
                    else:
                        res = await handler(payload) if inspect.iscoroutinefunction(handler) else handler(payload)
                except TypeError as te:
                    print(f"{_RELAY_DEBUG} receive: handler signature error for {case_name}: {te}")
                    raise
                except Exception as he:
                    print(f"{_RELAY_DEBUG} receive: handler error for case {case_name}: {he}")
                    import traceback
                    traceback.print_exc()
                    await self.send(text_data=json.dumps({
                        "type": case_name or data_type,
                        "status": {"state": "error", "code": 500, "msg": str(he)},
                        "data": {},
                    }, default=str))
                    handled = True
                    break
                res_type = res.get("type", "unknown") if isinstance(res, dict) else "unknown"
                print(f"{_RELAY_DEBUG} receive: sending response type={res_type}")
                return_data = json.dumps(res, default=str)
                await self.send(text_data=return_data)
                handled = True
                break

            if not handled:
                print(f"{_RELAY_DEBUG} receive: unknown command type: {data_type}")
        except Exception as e:
            print(f"{_RELAY_DEBUG} receive: error processing message: {e}")
            import traceback
            traceback.print_exc()
            try:
                await self.send(text_data=json.dumps({
                    "type": "ERROR",
                    "status": {"state": "error", "msg": str(e)},
                    "data": {},
                }, default=str))
            except Exception as send_err:
                print(f"{_RELAY_DEBUG} receive: could not send error response: {send_err}")


    async def _handle_start_sim_process(self, payload: dict):
        """
        Handles the START_SIM case.
        Delegates to Guard.main to fetch data, build graph, and compile pattern.
        Deactivates session upon success.
        """
        try:
            config = payload.get("data", {}).get("config", {})
            for k, v in config.items():
                try:
                    print(f"{_RELAY_DEBUG} _handle_start_sim_process: running guard.main for env_id={k}")
                    self.guard.main(env_id=k, env_data=v)
                    print(f"{_RELAY_DEBUG} _handle_start_sim_process: guard.main done for env_id={k}")
                except Exception as guard_err:
                    print(f"{_RELAY_DEBUG} _handle_start_sim_process: guard.main error for env_id={k}: {guard_err}")
                    import traceback
                    traceback.print_exc()
                    raise

                # send update user env
                try:
                    print(f"{_RELAY_DEBUG} _handle_start_sim_process: retrieving env table rows for user_id={self.user_id}")
                    updated_env = self.env_manager.retrieve_send_user_specific_env_table_rows(self.user_id)
                    await self.send(text_data=json.dumps({
                        "type": "GET_USERS_ENVS",
                        "data": updated_env
                    }, default=str))
                    print(f"{_RELAY_DEBUG} _handle_start_sim_process: sent GET_USERS_ENVS for env_id={k}")
                except Exception as ex:
                    print(f"{_RELAY_DEBUG} _handle_start_sim_process: error updating env status (continuing): {ex}")
                    import traceback
                    traceback.print_exc()

            # create new session
            if hasattr(self, "session_id") and self.session_id:
                try:
                    print(f"{_RELAY_DEBUG} _handle_start_sim_process: deactivating session {self.session_id}")
                    self.session_manager.deactivate_session(self.session_id)
                    self.session_id = self.session_manager.get_or_create_active_session(self.user_id)
                    print(f"{_RELAY_DEBUG} _handle_start_sim_process: new session_id={self.session_id}")
                except Exception as sess_err:
                    print(f"{_RELAY_DEBUG} _handle_start_sim_process: session deactivate/create error: {sess_err}")
                    import traceback
                    traceback.print_exc()

            response = {
                "type": "START_SIM",
                "status": {"state": "success", "code": 200, "msg": "Simulation started and session completed."},
                "data": {}
            }
            await self.send(text_data=json.dumps(response, default=str))
            await self.send_session()
            print(f"{_RELAY_DEBUG} _handle_start_sim_process: completed successfully")
        except Exception as e:
            print(f"{_RELAY_DEBUG} _handle_start_sim_process: error: {e}")
            import traceback
            traceback.print_exc()
            try:
                await self.send(text_data=json.dumps({
                    "type": "START_SIM",
                    "status": {"state": "error", "code": 500, "msg": str(e)},
                    "data": {}
                }))
            except Exception as send_err:
                print(f"{_RELAY_DEBUG} _handle_start_sim_process: could not send error response: {send_err}")



    async def batch_inject_env(self, payload):
        try:
            print(f"{_RELAY_DEBUG} batch_inject_env: parsing payload")
            data = payload.get("data", {})
            config = payload.get("config", {})
            if not config and "data" in payload and "config" in payload["data"]:
                config = payload["data"]["config"]
            elif not config and "config" in data:
                config = data["config"]
            if not config:
                print(f"{_RELAY_DEBUG} batch_inject_env: no config found in payload; skipping")
                return
            print(f"{_RELAY_DEBUG} batch_inject_env: processing {len(config)} env(s): {list(config.keys())}")

            for env_id, env_data in config.items():
                try:
                    print(f"{_RELAY_DEBUG} batch_inject_env: env_id={env_id}")
                    if not getattr(self, "world_creator", None) or not getattr(self.world_creator, "env_id_map", None):
                        print(f"{_RELAY_DEBUG} batch_inject_env: skipping invalid env_id (no world_creator.env_id_map): {env_id}")
                        await self.send(text_data=json.dumps({
                            "type": "deployment_error",
                            "admin_data": {"msg": f"skipping invalid env id: {env_id}"},
                        }))
                        continue
                    self.guard.sim_start_process(env_id, env_data)
                    await self.send(text_data=json.dumps({
                        "type": "deployment_success",
                        "admin_data": {"msg": f"Deployed machine to {env_id}"},
                    }))
                    print(f"{_RELAY_DEBUG} batch_inject_env: deployed env_id={env_id}")
                except Exception as e:
                    print(f"{_RELAY_DEBUG} batch_inject_env: deployment error for env_id={env_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    await self.send(text_data=json.dumps({
                        "type": "deployment_error",
                        "admin_data": {"msg": f"Failed to deploy machine to {env_id}: {str(e)}"},
                    }))
            print(f"{_RELAY_DEBUG} batch_inject_env: finished")
        except Exception as e:
            print(f"{_RELAY_DEBUG} batch_inject_env: error: {e}")
            import traceback
            traceback.print_exc()
            try:
                await self.send(text_data=json.dumps({
                    "type": "deployment_error",
                    "admin_data": {"msg": str(e)},
                }))
            except Exception as send_err:
                print(f"{_RELAY_DEBUG} batch_inject_env: could not send error response: {send_err}")



    async def disconnect(self, close_code):
        """Called when the websocket is disconnected."""
        try:
            print(f"{_RELAY_DEBUG} disconnect: close_code={close_code}, env_node={getattr(self, 'env_node', None) is not None}")
            if self.env_node is not None:
                print(f"{_RELAY_DEBUG} disconnect: env_node present; WebSocket disconnected with code: {close_code}")
        except Exception as e:
            print(f"{_RELAY_DEBUG} disconnect: error during cleanup: {e}")
            import traceback
            traceback.print_exc()




    async def error_response(self):
        try:
            print(f"{_RELAY_DEBUG} error_response: sending classification_error (invalid command)")
            await self.send(text_data=json.dumps({
                "type": "classification_error",
                "status": "success",
                "msg": "Invalid Command registered",
            }))
            print(f"{_RELAY_DEBUG} error_response: sent successfully")
        except Exception as e:
            print(f"{_RELAY_DEBUG} error_response: failed to send: {e}")
            import traceback
            traceback.print_exc()
            raise





"""

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

    # AIChat
    {"case": "CHAT", "desc": "Get Fields Params", "func": handle_start_chat, "req_struct": {"auth": {"user_id": "str", "field_id": "str"}}, "out_struct": {"type": "GET_FIELDS_PARAMS", "data": {"params": "list"}}},

                if data.get("files"):
                    # Classification step: use cases from file_manager.case (dynamic import)
                    print(f"{_RELAY_DEBUG} receive: CHAT with files -> file relay dispatch")
                    try:
                        mod = importlib.import_module("core.file_manager.case")
                        RELAY_FILE = getattr(mod, "RELAY_FILE", [])
                        payload_data = payload.get("data", {}) or {}
                        payload_auth = payload.get("auth", {}) or {}
                        if "user_id" not in payload_auth and self.user_id:
                            payload_auth["user_id"] = self.user_id
                            payload["auth"] = payload_auth
                        for case_info in RELAY_FILE:
                            case_name = case_info.get("case")
                            handler = case_info.get("func")
                            if not handler:
                                continue
                            req_struct = case_info.get("req_struct", {})
                            req_data = req_struct.get("data", {}) or {}
                            needs_files = "files" in req_data and bool(payload_data.get("files"))
                            needs_auth = not req_struct.get("auth") or (payload_auth.get("user_id") or self.user_id)
                            if needs_files and needs_auth:
                                print(f"{_RELAY_DEBUG} receive: matching file case {case_name}")
                                res = handler(payload) if not inspect.iscoroutinefunction(handler) else await handler(payload)
                                return_data = json.dumps(res, default=str)
                                await self.send(text_data=return_data)
                                print(f"{_RELAY_DEBUG} receive: file case {case_name} response sent")
                                return
                        print(f"{_RELAY_DEBUG} receive: no matching RELAY_FILE case for CHAT+files")
                        await self.send(text_data=json.dumps({
                            "type": "CHAT",
                            "status": {"state": "error", "msg": "No matching file case for CHAT+files"},
                            "data": {},
                        }, default=str))
                    except Exception as fe:
                        print(f"{_RELAY_DEBUG} receive: CHAT+files dispatch error: {fe}")
                        import traceback
                        traceback.print_exc()
                        await self.send(text_data=json.dumps({
                            "type": "CHAT",
                            "status": {"state": "error", "msg": str(fe)},
                            "data": {},
                        }, default=str))
                    return
                print(f"{_RELAY_DEBUG} receive: handling CHAT")
"""

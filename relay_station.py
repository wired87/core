import importlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx

from code_manipulation.graph_creator import StructInspector
from core.orchestrator_manager.orchestrator import OrchestratorManager


import asyncio

import json
import dotenv

from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

from urllib.parse import parse_qs

from predefined_case import RELAY_CASES_CONFIG

from utils.deserialize import deserialize

from chat_manger.main import AIChatClassifier

from utils.dj_websocket.handler import ConnectionManager


from graph.local_graph_utils import GUtils
from utils.utils import Utils


from core.session_manager.session import session_manager

from dataclasses import dataclass
from typing import Callable, Any, Awaitable, Dict, Optional


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
        self.cluster_acess_ip: int = None

        # Lazy: ConnectionManager, Utils, UserManager, cg, inspector, chat_classifier, tmp
        self._connection_manager = None
        self._utils = None
        self._user_manager = None
        self._cg = None
        self._inspector = None
        self._chat_classifier = None
        self._tmp = None

        self.file_store: list[str] = []
        self.qc = False
        self.demo_g_in_front = False
        self.start_up_path = "container/run.py"
        self.testing = True
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
        self.sim_start_puffer = 10
        self.demo = True

        self.required_steps = {
            "node_cfg": False,
            "world_cfg": False,
            "injection_cfg": False,
        }

        self.active_envs = {}
        self._root = None  # Set when tmp is first accessed

        self.worker_states = {"unknown": [], "error": [], "inactive": [], "active": []}
        self.possible_states = ["start", "check_status", "set_parameters", "stop"]

        self.session_id = None
        self.user_tables_authenticated = False
        self.cluster_auth_data = None

        self.con_type = "http" if os.name == "nt" else "https"
        self.cluster_domain = "127.0.0.1:8001" if os.name == "nt" else "clusterexpress.com"
        self.cluster_url = f"{self.con_type}://{self.cluster_domain}/"
        self.cluster_root = "http://127.0.0.1:8001" if self.testing else "cluster.botworld.cloud"

        self.auth_data = None
        self.relay_cases: list[RelayCase] = RELAY_CASES_CONFIG

        self._grid_streamer = None

        # Core components (g, qfu, guard, orchestrator) created in connect() when user_id is known
        self.g = GUtils(nx_only=False, G=nx.Graph(), g_from_path=None)
        self.orchestrator = None


    @property
    def connection_manager(self):
        if self._connection_manager is None:
            self._connection_manager = ConnectionManager()
        return self._connection_manager

    @property
    def utils(self):
        if self._utils is None:
            self._utils = Utils()
        return self._utils

    @property
    def user_manager(self):
        if self._user_manager is None:
            from core.managers_context import get_user_manager
            self._user_manager = get_user_manager()
        return self._user_manager

    @property
    def cg(self):
        if self._cg is None:
            self._cg = GUtils(nx_only=True, enable_data_store=False)
            self._inspector = StructInspector(self._cg.G)
            from core.handler_inspector import register_handlers_to_gutils
            register_handlers_to_gutils(self._cg)
        return self._cg

    @property
    def inspector(self):
        _ = self.cg  # Ensure cg (and inspector) initialized
        return self._inspector

    @property
    def chat_classifier(self):
        if self._chat_classifier is None:
            self._chat_classifier = AIChatClassifier(case_struct=self.relay_cases)
        return self._chat_classifier

    @property
    def tmp(self):
        if self._tmp is None:
            self._tmp = TemporaryDirectory()
            self._root = Path(self._tmp.name)
        return self._tmp

    @property
    def root(self):
        if self._root is None:
            _ = self.tmp
        return self._root if self._root is not None else Path(".")

    def consume_cases(self):
        pass  # TODO: implement case consumption

    def scan_dir_to_code_graph(self, root_dir: str = None) -> Dict[str, Any]:
        """
        Walk a local directory, scan all .py files, and convert each to graph format
        using self.inspector.convert_module_to_graph. Nodes are added to self.cg.G.

        Args:
            root_dir: Directory to scan. Defaults to project root (parent of relay_station).

        Returns:
            Dict with keys: scanned (int), converted (int), errors (list of {path, error}).
        """
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.normpath(root_dir)

        result = {"scanned": 0, "converted": 0, "errors": []}
        skip_dirs = {"__pycache__", ".git", "venv", ".venv", "node_modules"}

        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                filepath = os.path.join(dirpath, name)
                result["scanned"] += 1
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        code_content = f.read()
                except Exception as e:
                    result["errors"].append({"path": filepath, "error": str(e)})
                    continue

                rel = os.path.relpath(filepath, root_dir)
                module_name = rel[:-3].replace(os.sep, ".")

                try:
                    self.inspector.convert_module_to_graph(code_content, module_name)
                    result["converted"] += 1
                except Exception as e:
                    result["errors"].append({"path": filepath, "error": str(e)})

        return result

    async def connect(self):
        try:
            print(f"{_RELAY_DEBUG} connect: connection attempt registered")
            await self.accept()
            print(f"{_RELAY_DEBUG} connect: scope type = {self.scope.get('type')}")

            query_string = self.scope["query_string"].decode()
            query_params = parse_qs(query_string)
            print(f"{_RELAY_DEBUG} connect: query_params = {query_params}")

            received_key = (query_params.get("user_id") or [None])[0]
            resolved_user_id = await sync_to_async(self.user_manager.get_or_create_user)(
                received_key=received_key,
                email=None,
            )
            if not resolved_user_id:
                print(f"{_RELAY_DEBUG} connect: user resolution failed; declining")
                await self.close()
                return
            self.user_id = resolved_user_id
            print(f"{_RELAY_DEBUG} connect: user_id saved locally: {self.user_id}")

            def _create_orchestrator(cases, user_id, relay=None):
                return OrchestratorManager(cases, user_id=user_id, relay=relay)

            self.orchestrator = await sync_to_async(_create_orchestrator)(
                self.relay_cases,
                self.user_id,
                relay=self,
            )

            self._grid_streamer = None
            if os.getenv("GRID_STREAM_ENABLED", "false").lower() in ("true", "1"):
                from grid.streamer import GridStreamer
                async def _send_grid_frame(b: bytes):
                    await self.send(bytes_data=b)
                self._grid_streamer = GridStreamer(_send_grid_frame)
                self._grid_streamer.start()

            print(f"{_RELAY_DEBUG} connect: core components initialized")

            if not self.user_tables_authenticated:
                try:
                    print(f"{_RELAY_DEBUG} connect: running user_manager.initialize_qbrain_workflow")
                    user_email = None
                    workflow_results = await sync_to_async(self.user_manager.initialize_qbrain_workflow)(
                        uid=self.user_id,
                        email=user_email,
                    )
                    print(f"{_RELAY_DEBUG} connect: user workflow completed: {workflow_results}")
                except Exception as e:
                    print(f"{_RELAY_DEBUG} connect: user workflow error (continuing): {e}")
                    import traceback
                    traceback.print_exc()
                self.user_tables_authenticated = True

            session_id = await self._resolve_session()
            if session_id is None:
                print(f"{_RELAY_DEBUG} connect: session resolution failed for user {self.user_id}; declining")
                await self.close()
                return
            self._save_session_locally(session_id)

            print(f"{_RELAY_DEBUG} connect: sending session and user sessions to client")
            await self.send_session()
            await self._send_all_user_sessions()
            print(f"{_RELAY_DEBUG} connect: request for user {self.user_id} ACCEPTED")
        except Exception as e:
            print(f"{_RELAY_DEBUG} connect: FATAL error: {e}")
            import traceback
            traceback.print_exc()
            try:
                await self.close()
            except Exception as close_err:
                print(f"{_RELAY_DEBUG} connect: error during close: {close_err}")


    async def _resolve_session(self) -> Optional[int]:
        """
        Resolve session for current user: fetch active or create new.
        Returns valid session_id (int) or None on failure.
        """
        return await sync_to_async(session_manager.get_or_create_active_session)(self.user_id)

    def _save_session_locally(self, session_id: int) -> None:
        """Store valid session_id on Relay instance for the connection lifecycle."""
        if session_id is None or not isinstance(session_id, (int, str)) or session_id == "":
            return
        self.session_id = int(session_id)
        print(f"{_RELAY_DEBUG} _save_session_locally: saved session_id={self.session_id}")

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

    async def _send_all_user_sessions(self) -> None:
        """Fetch all sessions for the current user and send them to the client."""
        try:
            sessions = await sync_to_async(session_manager.list_user_sessions)(self.user_id)
            payload = {
                "type": "LIST_USERS_SESSIONS",
                "auth": {"user_id": self.user_id, "sid": getattr(self, "session_id", None)},
                "data": {"sessions": sessions},
            }
            await self.send(text_data=json.dumps(payload, default=str))
            print(f"{_RELAY_DEBUG} _send_all_user_sessions: sent {len(sessions)} session(s)")
        except Exception as e:
            print(f"{_RELAY_DEBUG} _send_all_user_sessions: error: {e}")
            import traceback
            traceback.print_exc()



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
            #print(f"{_RELAY_DEBUG} receive: raw text_data length={len(text_data) if text_data else 0)")
            payload = deserialize(text_data)
            data_type = payload.get("type")
            #print(f"{_RELAY_DEBUG} receive: type={data_type}")

            if self.session_id:
                if "auth" not in payload:
                    payload["auth"] = {}
                if "session_id" not in payload["auth"]:
                    payload["auth"]["session_id"] = self.session_id

            # Orchestrator is now the primary classifier/dispatcher for all relay cases,
            # including CHAT and START_SIM (via injected helpers).
            orchestrator_response = None
            try:
                orchestrator_response = await self.orchestrator.handle_relay_payload(
                    payload=payload,
                    user_id=self.user_id,
                    session_id=str(self.session_id) if self.session_id else None,
                )
            except Exception as orch_err:
                print(f"{_RELAY_DEBUG} receive: orchestrator error: {orch_err}")
                import traceback
                traceback.print_exc()

            if isinstance(orchestrator_response, dict):
                # Special sentinel for side-effect-only handling (e.g. START_SIM)
                if orchestrator_response.get("__handled_by_side_effect__"):
                    print(f"{_RELAY_DEBUG} receive: orchestrator handled {data_type} via side effects")
                    return

            if isinstance(orchestrator_response, list):
                orchestrator_response
                await asyncio.gather(
                    *[
                        self.send_message(item)
                        for item in orchestrator_response
                    ]
                )
                return

            if isinstance(orchestrator_response, dict):
                await self.send_message(orchestrator_response)

            print(f"{_RELAY_DEBUG} receive: unknown command type (unhandled): {data_type}")
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


    async def send_message(self, orchestrator_response):
        res_type = orchestrator_response.get("type", "unknown")
        print(f"{_RELAY_DEBUG} receive: orchestrator handled type={res_type}")
        return_data = json.dumps(orchestrator_response, default=str)
        await self.send(text_data=return_data)



    async def disconnect(self, close_code):
        """Called when the websocket is disconnected."""
        try:
            if getattr(self, "_grid_streamer", None) is not None:
                self._grid_streamer.stop()
                self._grid_streamer = None
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


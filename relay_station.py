import os
import threading
import time
import requests
import json
import dotenv
from channels.generic.websocket import AsyncWebsocketConsumer

import asyncio
from urllib.parse import parse_qs

from create_env import EnvCreatorProcess
from fb_core.real_time_database import FBRTDBMgr

from _chat.log_sum import LogAIExplain
from bm.settings import TEST_ENV_ID, TEST_USER_ID
from workflows.create_ws_prod import WorldCreationWf
from workflows.data_distirbutor import DataDistributor
from workflows.deploy_sim import GcpDockerVmDeployer
from workflows.node_cfg_manager import NodeCfgManager
from utils.deserialize import deserialize

dotenv.load_dotenv()

from _chat.main import AIChatClassifier

from utils.dj_websocket.handler import ConnectionManager
from utils.get_local_ip import get_local_ip

from utils.graph.local_graph_utils import GUtils
from utils.id_gen import generate_id
from utils.utils import Utils


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
        #self.sim = SimCore()
        self.utils = Utils()

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

        self.ws_handler = None
        self.external_vm = False
        self.sim_ready = False
        self.sim_start_puffer = 10  # seconds to wait when rcv start trigger
        self.demo = True


        self.required_steps = {
            "node_cfg": False,
            "world_cfg": False,
        }

        self.active_envs = {}

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

        self.sim_deployer = GcpDockerVmDeployer()


        if self.testing is True:
            self.cluster_root = "http://127.0.0.1:8001"
        else:
            self.cluster_root = "cluster.botworld.cloud"

        self.auth_data = None

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

        # todo collect more sim data like len, elements, ...
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
            user_id=self.user_id,
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

        # self.qf_utils = QFUtils(self.g)

        print("request accepted")
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

            if data_type == "world_cfg":
                print("CREATE WORLD REQUEST RECEIVED")
                await self.world_creator.world_cfg_process(data["world_cfg"])

            elif data_type == "node_cfg":
                print("CREATE NODE CFG REQUEST RECEIVED")
                self.world_creator.node_cfg_process(data)

            elif data_type == "get_envs":
                await self.send(
                    text_data=json.dumps({
                        "type": "env_ids",
                        "data": self.db_manager.get_child(
                            path=f"users/{self.user_id}/env/"
                        ),
                    })
                )

            elif data_type == "get_data":
                await self.send(
                    text_data=json.dumps({
                        "type": "data",
                        "data": self.db_manager.get_data(path=f"users/{self.user_id}/env/{data.get('env_id')}/datastore/"),
                    })
                )

            elif data_type == "start_sim":
                await self.handle_sim_start(data)

            else:
                print(f"Unknown command type received: {data_type}")

        except Exception as e:
            print(f">>Error processing received message: {e}")



    async def handle_sim_start(self, data):
        print("START SIM REQUEST RECEIVED")
        self.env_creator = EnvCreatorProcess(self.user_id)

        env_ids = data.get("data", {}).get("env_ids")
        for env_id in env_ids:
            try:
                if self.world_creator.env_id_map:
                    self.sim_deployer.create_vm(
                        instance_name=env_id,
                        gpu_count=1,
                        testing=self.testing,
                        env=self.env_creator.create_env_variables(env_id),
                    )
                    await self.send(
                        text_data=json.dumps({
                            "type": "deployment_success",
                            "data": {
                                "msg": "Invalid Command registered",
                            },
                        })
                    )
                else:
                    print(f"skipping invalid env id: {env_id}")
                    await self.send(
                        text_data=json.dumps({
                            "type": "deployment_error",
                            "data": {
                                "msg": f"skipping invalid env id: {env_id}",
                            },
                        }))
            except Exception as e:
                print(f"Err deploymnt: {e}")
                await self.send(
                    text_data=json.dumps({
                        "type": "deployment_success",
                        "data": {
                            "msg": f"Deployed machine to {env_id}",
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
            "data": response
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
                    "data": self.created_envs,
                }
            )
        )




    async def comand_handler(
            self,
            data:list[str],
    ):
        """
        Deploy a docker in created vm and executes
        """
        classification = self.chat_classifier._classify_input(
            user_input=data.get("text")
        )
        # jeder node horcht auf state changes (zb start = active etc) von globalem store

        if classification in self.possible_states:
            data: dict = {
                "state": classification
            }
            if classification == "start":
                if self.demo is True:
                    await self.demo_workflow()
                else:
                    # send local docker start request

                    c_data = {  # InboundPayload
                        "data": {
                            "type": classification
                        },
                        "type": "state_change",
                    }
                    # START SIM
                    for env_id in list(self.env_cfg.keys()):
                        self.utils.apost_gather(
                            url=f"{self.cluster_root}/{env_id.replace('_','-')}"
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
            "data": {
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
                    print(f"Error wile reuqest bq data: {e}")
                    time.sleep(5)

        def handle_data(data):
            asyncio.run(self.handle_data_response(data))

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
            "data": data
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
            "data": {
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
    #ihonicy

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
            "data": {
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
            {"data": self.sim.updator.datastore}
        )
        return

    async def handle_cluster_command(self, c_data):
        if getattr(self, "cluster_auth_data", None) is None:
            data = {
                "type": "auth",
                "data": {
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
            "data": content
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
        # data => {'type': None, 'path': '/', 'data': {'F_mu_
        print("handle_data_changes")
        # todo make a class for it
        all_subs = self.qf_utils.get_all_subs_list(just_id=True)

        attrs = data["data"]
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
                    "data": {
                        "id": nid,
                        "data": data,
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
                        "data": {
                            "data": data,
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
                                "data": {
                                    "id": nid,
                                    "data": data,
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



"""

 def _connect(self, target_ip):
        connect_attempts = 0
        max_attempts = 20

        while not self.websocket_connection and connect_attempts < max_attempts:
            try:
                print(
                    f"Versuche, WebSocket-Verbindung zu ws://{target_ip}:{self.ws_port} herzustellen... (Versuch {connect_attempts + 1})")
                self.websocket_connection = websocket.create_connection(
                    f"{self.ws_type}://{target_ip}:{self.ws_port}")
                print(f"Verbindung zu {self.ws_type}://{target_ip}:{self.ws_port} hergestellt.")
                self.websocket_connection.send(
                    text_data=json.dumps(
                        {
                            "type": "init_hs_relay",
                            "session_id": self.session_id,
                            "key": self.env_id
                        }
                    )
                )
            except Exception as e:
                print(f"Verbindungsfehler: {e}. Warte 5 Sekunden...")
                time.sleep(5)
                connect_attempts += 1





            elif data_type == "env_data":
                await self.data_distributor.send_data(data)
 

            elif data_type == "ai_log_sum":
                await self.ai_log_sum_process(data)

            elif data_type == "rcv_session_id":
                # receive ids form already existing sessions.
                # Init G
                sessions:list[str] = data.get("session_ids")

            elif data_type == "logs":
                await self.log_request_handler(data)

            elif data_type == "COMMAND":
                await self.comand_handler(data)



"""
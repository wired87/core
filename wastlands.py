"""


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

            setattr(self, "cluster_auth_data", res_data)
        elif "type" in res_data and res_data["type"] == "status_success_distribution":
            print(f"response of command distribution received: {res_data}")
            await self.send(text_data=json.dumps({
                "type": "distribution_complete",
                "status": "success",
            }))
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

    async def _handle_convert_module(self, payload):

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





async def check_ready(self, env_ids: list[str]):
    print("Start ready Thread")

    def _connect():

        ready_envs: list = []
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
"""


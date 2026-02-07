import json


class NodeCfgManager:

    def __init__(
            self,
            user_id,
            cluster_url,
            utils,
            parent,
    ):
        #
        self.parent=parent
        self.cluster_url=cluster_url
        self.user_id=user_id
        self.utils=utils
        self.cfg_store = {}
        print("nodecfg manager inisitlized")

    async def node_cfg_process(self, data):
        print("Handling node process")
        env_id = data.get("env_id")
        node_cfg = data.get("node_cfg")
        nid = data.get("nid")

        global_struct: dict = self.check_env_authenticated(env_id)

        # Check ENV alive
        if global_struct["authenticated"] is True and global_struct["cfgs_created"] is True and global_struct["ready"] is True:

            if node_cfg is not None and env_id is not None:
                request_endp = f"{self.cluster_url}/{env_id}/root/"
                paylaod = {
                    "type": "node_cfg",
                    "cfg": node_cfg,
                }
                response = await self.utils.apost(
                    url=request_endp,
                    data=paylaod
                )
                print(f"Node cfg distribtuted. response: {response}")

                self.update_node_cfg_glob_state(env_id)

            await self.parent.send(
                text_data=json.dumps(
                    {
                        "type": "node_cfg",
                        "status": "successful upserted node cfg",
                    }
                )
            )
        else:
            print("Env not ready now. Upsert cg and save trigger")

            self.upsert_node_cfg(env_id, nid, node_cfg)

            if env_id not in self.cfg_store:
                self.cfg_store[env_id] = []

            self.cfg_store[env_id].append(nid)

            await self.parent.send(
                text_data=json.dumps(
                    {
                        "type": "message",
                        "admin_data": "Upserted admin_data in store",
                        "status": "TEMPORARY_SAVED",
                    }
                )
            )


    def check_env_authenticated(self, env_id):
        global_struct:dict = self.db_manager.get_data(
            path=f"users/{self.user_id}/env/{env_id.replace('-', '_')}/global_states/",
        )
        return global_struct

    def upsert_node_cfg(self, env_id, nid, node_cfg):
        self.db_manager.upsert_data(
            path=f"users/{self.user_id}/env/{env_id.replace('-', '_')}/cfg/{nid}/",
            data=node_cfg
        )
        print("NodeCfg upserted")

    def update_node_cfg_glob_state(self, env_id):
        self.db_manager.upsert_data(
            path=f"users/{self.user_id}/env/{env_id.replace('-', '_')}/global_states/",
            data={
                "min_node_cfg_created": True
            }
        )
        print(f"min node struct updated")


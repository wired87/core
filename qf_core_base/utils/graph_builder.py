import os

from _ray_core.ray_validator import RayValidator
from app_utils import ENV_ID
from fb_core.real_time_database import FirebaseRTDBManager
from qf_core_base.qf_utils.all_subs import ALL_SUBS


class GraphBuilder:
    def __init__(self, user_id, g, ray_validator=None, testing=False, host=None, db_manager=None):
        self.ray_validator = ray_validator or RayValidator()
        self.db_manager = db_manager or FirebaseRTDBManager()
        self.user_id = user_id
        self.g = g
        self.host = host
        self.testing = testing

    async def main(
            self,
            env_ids:list[str],
            build_frontend_data=False,
            include_metadata=False,
            reset_g_after=True,
            testing=False
    ):
        print("Handling DB Stuff")
        content = {}
        env_listener_paths = {}

        for env_id in env_ids:
            # Set db urls
            self.database = f"users/{self.user_id}/env/{env_id}"
            self.global_states_path = f"{self.database}/global_states/"
            self.worker_states_db_path = f"{self.database}/metadata/"
            self.instance = os.environ.get("FIREBASE_RTDB")

            env = await self.build_graph(testing)

            # EXTEND WITH METADATA
            if include_metadata is True:
                self.metadata_process()

            if build_frontend_data is True:
                stuff, listener_paths = await self.create_frontend_env_content()

                env_listener_paths[env_id] = listener_paths
                content[env_id] = stuff
            else:
                content[env_id] = env
            if reset_g_after is True:
                self.ray_validator.call(
                    method_name="set_G",
                    G=None
                )
            print(f"DB Stuff for {env_id} set")
        return content, env_listener_paths



    async def create_frontend_env_content(self):
        nodes = {}
        edges = {}
        id_map = set()

        for nid, attrs in self.g.G.nodes(data=True):
            if attrs.get("type").lower() not in ["users", "user"]:
                nodes[nid] = {
                        "id": nid,
                        "pos": attrs.get("pos"),
                        "meta": attrs.get("meta"),
                        "color": "#004999",
                        "logs": {}
                    }

                id_map.add(nid)
        print("Nodes extracted", len(nodes.keys()))

        for src, trgt, attrs in self.g.G.edges(data=True):
            if (attrs.get("src_layer").lower() not in ["env", "user", "users"]
                and attrs.get("trgt_layer").lower() not in ["env", "user", "users"]):
                edges[attrs["id"]] = {
                    "src": src,
                    "trgt": trgt,
                }
        print("Edges extracted", len(edges.keys()))

        # EXTRACT PATHS
        all_paths = self.db_manager._get_db_paths_from_G(
            g=self.g,
            db_base=self.database,
        )

        for nid, attrs in [(nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs["type"] in [*ALL_SUBS, "ENV"]]:
            ntype = attrs['type']

            value_path = f"{self.database}/{ntype}/{nid}/{self.g.qf_utils.get_field_key(ntype)}"

            all_paths["value"].append(value_path)

        env_content = {
            "edges": edges,
            "nodes": nodes,
        }
        return env_content, all_paths




    async def build_graph(self, db_root=None, testing=False):
        db_root=db_root or self.database
        if testing is True:
            if os.path.isfile(self.g.demo_G_save_path) is True:
                print("Build Demo G")
                self.g.load_graph(local_g_path=self.g.demo_G_save_path)
        else:
            initial_data = self.db_manager._fetch_g_data(
                db_root=db_root
            )

            # Build a G from init data and load in self.g
            env, env_id = self.ray_validator.call(
                method_name="build_G_from_data",
                initial_data=initial_data,
                env_id=ENV_ID,
                save_demo=False,
            )
            return env





    def metadata_process(self):
        data = self.db_manager.get_data(
            path=f"{self.database}/metadata/"
        )

        all_sub_nodes: list[str] = self.g.get_nodes(
            filter_key="type",
            filter_value=ALL_SUBS,
            just_id=True,
        )

        print("metadata received:", data.keys())
        if data:
            # print("data[metadata]", data["metadata"])
            for node_id, metadata in data["metadata"].items():
                meta_of_interst = {k: v for k, v in metadata.items() if k not in ["id"]}
                print(f"add {meta_of_interst} to {node_id}")

                # NODE FROM ALL_SUBS?
                node_valid:bool = self.validate_meta_nid(
                    node_id,
                    all_sub_nodes,
                )
                if node_valid is True or (self.testing and node_id == ENV_ID):
                    continue
                    # todo include extra helper node (logs)
                self.g.G.nodes[node_id]["meta"] = meta_of_interst




    def validate_meta_nid(self, node_id, all_sub_nodes):
        """
        Extend the hlper ndoe list with all helper
        nodes to exclude in the graph build up
        (todo helper nodes get accessible through the worker node)
        """
        if node_id in all_sub_nodes:
            return True
        return False



    def build_demo_G(self):
        if self.testing is True:
            demo_G_save_path = self.ray_validator.call(
                    method_name="get_demo_G_save_path"
                )

            if demo_G_save_path is not None:
                if os.path.isfile(demo_G_save_path) is True and os.name == "nt":
                    print("Build Demo G from path:", demo_G_save_path)
                    env, env_id = self.ray_validator.call(
                        method_name="load_graph",
                        local_g_path=demo_G_save_path
                        )
                    print("Demo G build successful")
                    return env, env_id


    def save_cfg(
            self,
            cfg_struct, #app_name: worldcfg,...
    ):
        """
        SAVE CFG IN ENV CFG STRUCT
        """
        print("Save GKE cfg for depl & service")
        for app_name, struct in cfg_struct.items():
            try:
                db_root = f"users/{self.user_id}/env/{app_name}"
                # create enw space for env cfg
                path = f"{db_root}/cfg/world/"

                # UPSERT ALL CFG STRUCTS FOR ENV
                data = self.db_manager._check_keys(
                    struct
                )

                if "env" in data:
                    data.pop("env")

                self.db_manager.upsert_data(
                    path=path,
                    data=data
                )

                print(f"Resource CFG for {app_name} upserted")
            except Exception as e:
                print(f"Err save_gke_cfg: {e}")
        print("Save resources cfg process finished")




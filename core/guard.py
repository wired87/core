import json
import logging
import pprint
import time
from datetime import datetime
from typing import Any, Dict

import networkx as nx

from _god.create_world import God
from core._ray_core.globacs.state_handler.main import StateHandler

from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin
from core.app_utils import SCHEMA_GRID
from core.env_manager import EnvManager
from core.fields_manager.fields_lib import FieldsManager, generate_numeric_id
from core.injection_manager import InjectionManager
from core.module_manager.ws_modules_manager import ModuleWsManager
from core.user_manager import UserManager

from gnn_master.pattern_store import GStore
from core.module_manager.mcreator import ModuleCreator
from qf_utils.all_subs import ALL_SUBS
from qf_utils.qf_utils import QFUtils
from workflows.deploy_sim import DeploymentHandler

class PatternMaster:

    def __init__(self, g):
        self.g=g
        self.modules_struct=[]



    def get_empty_field_structure(self):
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        self.modules_struct = [[] for _ in range(len(modules))]

        for i, (mid, m) in enumerate(modules):
            # get module fields
            fields = self.g.get_neighbor_list(
                node=mid,
                target_type="FIELD",
            )
            field_struct = [[] for _ in range(len(fields))]

            # SET EMPTY DIMS FOR EACH FIELD
            self.modules_struct[i] = field_struct
        return self.modules_struct



class Guard(
    StateHandler,
    God,
    PatternMaster,
):
    # todo answer caching
    # todo cross module param edge map
    """
    nodes -> guard: extedn admin_data
    """

    def __init__(
        self,
        qfu,
        g,
        user_id,
    ):
        print("Initializing Guard...")
        super().__init__()
        God.__init__(
            self,
            g,
            qfu,
        )
        PatternMaster.__init__(
            self,
            g,
        )
        self.user_id=user_id
        self.deployment_handler = DeploymentHandler(
            user_id
        )

        # DB MANAGERS
        self.module_db_manager = ModuleWsManager()
        self.field_manager = FieldsManager()
        self.injection_manager = InjectionManager()
        self.env_manager = EnvManager()
        self.user_manager = UserManager()


        self.world_cfg=None
        self.artifact_admin = ArtifactAdmin()

        self.time = 0

        self.qfu:QFUtils = qfu
        self.g = g
        self.ready_map = {
            k: False
            for k in ALL_SUBS
        }

        self.mcreator = ModuleCreator(
            self.g.G,
            self.qfu,
        )

        self.pattern_arsenal = GStore

        # Time Series Model Init
        self.prev_state = None
        self.model_params = None
        self.fnished_modules = False

        self.fields = []

        print("Guard Initialized!")



    def sim_start_process(self, payload):
        """
        1. Parse payload to get IDs.
        2. Batch fetch data.
        3. Convert to pattern (Graph).
        4. Compile pattern.
        """
        import pprint
        
        data = payload.get("data", {})
        config = payload.get("config", {})
        if not config and "data" in payload and "config" in payload["data"]:
             config = payload["data"]["config"]
        elif not config and "config" in data:
             config = data["config"]
             
        envs_config = config.get("envs", {})
        
        # 1. Collect IDs
        env_ids = set()
        module_ids = set()
        field_ids = set()
        inj_ids = set()
        
        for env_id, env_data in envs_config.items():
            env_ids.add(env_id)
            modules = env_data.get("modules", {})
            for mod_id, mod_data in modules.items():
                module_ids.add(mod_id)
                fields = mod_data.get("fields", {})
                for field_id, field_data in fields.items():
                    field_ids.add(field_id)
                    injections = field_data.get("injections", {})
                    for pos, inj_id in injections.items():
                        inj_ids.add(inj_id)

        # 2. Batch Fetch
        print(f"Fetching data: {len(env_ids)} Envs, {len(module_ids)} Modules, {len(field_ids)} Fields, {len(inj_ids)} Injections")
        
        fetched_envs = {}
        if env_ids:
            res = self.env_manager.retrieve_env_from_id(list(env_ids))
            fetched_envs = {item["id"]: item for item in res.get("envs", [])}
            
        fetched_modules = {}
        if module_ids:
            res = self.module_db_manager.get_module_by_id(list(module_ids))
            fetched_modules = {item["id"]: item for item in res.get("modules", [])}

        fetched_fields = {}
        if field_ids:
            res = self.field_manager.get_fields_by_id(list(field_ids))
            fetched_fields = {item["id"]: item for item in res} 
            
        fetched_injections = {}
        if inj_ids:
            res = self.injection_manager.get_inj_list(list(inj_ids))
            fetched_injections = {item["id"]: item for item in res}

        # 3. Populate Graph (Pattern Construction)
        for env_id, env_config in envs_config.items():
            env_data = fetched_envs.get(env_id)
            if env_data:
                try:
                    params = self.qfu.create_env(world_cfg=env_data)
                except Exception as e:
                    print(f"Error create_env for {env_id}: {e}")
                    params = {}
                
                self.g.add_node(dict(
                    nid=env_id,
                    type="ENV",
                    **env_data
                ))
            else:
                self.g.add_node({"nid": env_id, "type": "ENV"})

            modules_config = env_config.get("modules", {})
            for mod_id, mod_config in modules_config.items():
                mod_data = fetched_modules.get(mod_id)
                if mod_data:
                    self.g.add_node(dict(
                        nid=mod_id,
                        type="MODULE",
                        **mod_data
                    ))
                else:
                    self.g.add_node({"nid": mod_id, "type": "MODULE"})

                self.g.add_edge(env_id, mod_id, attrs={"rel": "contains"})

                fields_config = mod_config.get("fields", {})
                for field_id, field_config in fields_config.items():
                    field_data = fetched_fields.get(field_id)
                    if field_data:
                        params = field_data.get("params", {})
                        self.g.add_node(dict(
                            nid=field_id.upper(),
                            type="FIELD",
                            field_keys=list(params.keys()),
                            value=list(params.values()),
                            axis_def=self.qfu.set_axis(params),
                            **field_data
                        ))
                        try:
                           self.qfu.add_params_link_fields(params, field_id, mod_id)
                        except Exception as e:
                            print(f"Error linking params for field {field_id}: {e}")

                    else:
                        self.g.add_node({"nid": field_id, "type": "FIELD"})

                    self.g.add_edge(mod_id, field_id, attrs={"rel": "has_finteractant"})
                    
                    injections_config = field_config.get("injections", {})
                    for pos, inj_id in injections_config.items():
                        inj_data = fetched_injections.get(inj_id)
                        if inj_data:
                            self.g.add_node(dict(
                                nid=inj_id,
                                type="INJECTION",
                                **inj_data
                            ))
                        else:
                            self.g.add_node({"nid": inj_id, "type": "INJECTION"})
                            
                        self.g.add_edge(field_id, inj_id, attrs={"rel": "injection_at", "pos": pos})

        print("Graph populated with fetched data.")

        # 4. Compile Pattern
        try:
            self.main()
            print("Pattern Compiled:")
            pprint.pprint(self.module_pattern_collector)
        except Exception as e:
             print(f"Error in compile_pattern: {e}")
             import traceback
             traceback.print_exc()












    def main(self, env_ids:list[str]):
        """
        CREATE/COLLECT PATTERNS FOR ALL ENVS AND CREATE VM
        """
        # pattern extractor -> include all fields
        # need
        # inj
        # wcfg
        self.compile_pattern()
        self.create_db()
        self.set_param_pathway()

        for env in env_ids:
            inj_pattern = self.g.get_neighbor_list(
                node=env,
                target_type="INJECTION_PATTERN",
            )

            # CREATE VM PAYLOAD
            docker_payload = {
                "UPDATOR_PATTERN": self.module_pattern_collector,
                "DB": self.db,
                "AMOUNT_NODES": self.amount_nodes,
                "INJECTION_PATTERN": inj_pattern,
                "WORLD_CFG": json.dumps(self.world_cfg),
                "ENERGY_MAP": self.module_map_entry
            }
            print(f"DOCKER ENV VARS CREATED FOR {env}:")
            pprint.pp(docker_payload)

            # PUBLISH VM
            self.deploy_sim(
                env,
                env_var_cfg=docker_payload
            )

        print(f"Deployment of {env_ids} Finished!")


    def get_state(self):
        return len(self.fields) and self.world_cfg


    def set_inj_pattern(
            self,
            inj_pattern:dict[
                str, dict[tuple, list[list, list]]
            ], # pos:time,val -> entire sim len captured
            env_id:str
    ):
        """
        frontend -> relay-> inj_pattern
        self.inj_pattern = [
            [pos, [[t],[e]]],
        ]
        """
        # retrieve empty struct based on all fieds and modules creatde
        inj_struct = self.get_empty_field_structure()

        for ntype, pos_inj_struct in inj_pattern.items():
            fattrs = self.g.G.nodes[ntype]
            module_index: int = fattrs["module_index"]
            field_index:int = fattrs["field_index"]
            for pos, inj_pattern in pos_inj_struct:
                inj_struct[
                    module_index
                ][
                    field_index
                ][
                    SCHEMA_GRID.index(pos)
                ] = inj_pattern

        # INJECTION_STRUCT -> G
        inj_id = f"inj_{env_id}"
        self.g.add_node(
            dict(
                nid=inj_id,
                pattern=inj_struct,
                type="INJECTION_PATTERN",
            )
        )

        # ENV -> INJECTION_STRUCT
        self.g.add_edge(
            src=env_id,
            trgt=inj_id,
            attrs=dict(
                src_layer="ENV",
                trgt_layer="INJECTION_PATTERN",
                rel="has_injection_pattern",
                # **edge_yaml_cache[edge_key],
            )
        )
        print("set_inj_pattern finished")


    def set_wcfg(self, world_cfg):
        self.world_cfg = world_cfg

        amount_nodes = world_cfg["amount_nodes"]

        schema_grid = [
            (x, y, z)
            for x in range(amount_nodes)
            for y in range(amount_nodes)
            for z in range(amount_nodes)
        ]

        node = {
            "nid": world_cfg["env_id"],
            "cfg": world_cfg,
            "params": self.create_env(world_cfg),
            "schema_grid": schema_grid,
            "type": "ENV",
        }
        self.g.add_node(node)
        return node




    def handle_module_ready(
            self,
            mid,
            field_ids
    ):
        """
        Trigger when module was created and fields initialized
        runnable alread include
        """
        try:
            self.g.G.nodes[mid]["ready"] = True
            self.g.G.nodes[mid]["field_keys"] = field_ids
        except Exception as e:
            print("Err receive_ready", mid, e)

    def set_param_pathway(
            self,
            trgt_key="energy"
    ) -> list[list[int]]:
        # exact same format
        print("set_param_pathway started")
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        self.module_map_entry = [
            [] for _ in range(len(modules))
        ]

        for mid, mattrs in modules:
            # get param (method) neighbors
            # get params
            # extract index
            mindex = mattrs["module_index"]
            fields:list[tuple] = self.g.get_neighbor_list(
                node=mattrs["nid"],
                target_type="FIELD"
            )
            energy_param_map=[
                None
                for _ in range(len(fields))
            ]
            for fid, fattrs in fields:
                try:
                    energy_param_map[fattrs["field_index"]
                    ] = fattrs["field_keys"].index(
                        trgt_key
                    )
                except Exception as e:
                    print(f"Err map pathway to param:{trgt_key}: {e}")
            self.module_map_entry[mindex] = energy_param_map
        print(f"param map to {trgt_key} build")


    def deploy_sim(self, env_id, env_var_cfg):
        # DEPLOY
        container_env = self.deployment_handler.env_creator.create_env_variables(
            env_id=env_id,
            cfg=env_var_cfg
        )
        self.deployment_handler.create_vm(
            instance_name=env_id,
            testing=self.testing,
            image=self.artifact_admin.get_latest_image(),
            container_env=container_env
        )
        print("deploy_sim finsihed")


    def set_param_edge_pattern(self):
            
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        for m in modules:
            # get param (method) neighbors
            # get params
            # extract index
            module_index = m["module_index"]
            fields = self.g.get_neighbor_list(
                node=m["nid"],
                target_type="PARAM"
            )

            tasks = [
                get_actor(f).get_runnable_pattern.remote(
                )
                for f in fields
            ]
            ray.get(tasks)
        return


    def set_param_index_map(self):
        # create param index map
        pindex_map_fields = [
            [] for _ in range(
            len(self.fields))
        ]

        # get all modules
        fields = self.g.get_nodes(
            filter_key="type",
            filter_value="FIELD",
        )

        # todo sync finish state of all
        #  fields before
        for fid, f in fields:
            arsenal_struct = f["arsenal_struct"]
            param_map = arsenal_struct["params"]
            param_index_map = [[], []]
            for param in param_map:
                # param: str
                if param in f["keys"]:
                    # first param mapping for index
                    param_index_map[0].append(None)
                    param_index_map[1].append(
                        f["keys"].index(param)
                    )

                pindex_map_fields[
                    self.fields.index(fid)
                ] = param_index_map

                self.g.update_node(
                    dict(
                        fid,
                        param_index_map=param_index_map
                    )
                )
            print("create_param_index_map finisehd:", )
            return pindex_map_fields






    def create_injector(self):

        from injector import Injector
        self.injector = Injector.options(
            name="INJECTOR",
            lifetime="detached"
        ).remote(
            world_cfg=self.world_cfg
        )

        self.g.add_node(
            dict(
                nid="INJECTOR",
                ref=self.injector._ray_actor_id.binary().hex(),
                type="ACTOR",
            )
        )

        self.await_alive(["INJECTOR"])
        print("create_injector")




    def create_and_distribute_data(self):
        """
        Loop modules -> fields -> admin_data(params)
        write param to nnx.Module
        """

        # create pattern
        modules:list[tuple] = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        for mid, mttrs in modules:
            # get module
            fields = self.g.get_neighbor_list(
                node=mid,
                target_type="FIELD",
            )

            for fid, fattrs in fields:
                params:dict[str, Any] = fattrs["admin_data"]

                for i, (k, v) in enumerate(params.items()):
                    # get param module
                    param = self.g.G.nodes[k]
                    module = param["module"]

                    # add admin_data
                    module.add_data(
                        v,
                        self.amount_nodes
                    )

        print("Params sorted to nodes")
        return


    def all_nodes_ready(self, trgt_types) -> bool:
        return all(
            attrs["ready"] is True
            for attrs in list(
                self.g.G.nodes[ntype]
                for ntype in trgt_types
            )
        )

    def handle_mod_stack(
            self,
            root,
    ):
        """
        receive files
        write to temp
        load in ModuleCreator -> create modules
        """
        print("handle_mod_stack")
        # create modules form files -> save converted in G
        self.mcreator.main(
            temp_path=root
        )
        print("handle_mod_stack finished successfully")

    def convert_config_to_graph(self, input_struct: dict):
        """
        Convert provided input structure to nx.Graph.
        Node types: ENV, MODULE, FIELD, INJECTION.
        Logical linking fields together.
        """
        import networkx as nx
        
        G = nx.Graph()
        
        config = input_struct.get("config", {})
        envs = config.get("envs", {})

        for env_id, env_data in envs.items():
            # Add ENV node
            if not G.has_node(env_id):
                G.add_node(env_id, type="ENV")
            
            modules = env_data.get("modules", {})
            for mod_id, mod_data in modules.items():
                # Add MODULE node
                if not G.has_node(mod_id):
                    G.add_node(mod_id, type="MODULE")
                
                # Link ENV -> MODULE
                G.add_edge(env_id, mod_id, rel="contains")
                
                fields = mod_data.get("fields", {})
                for field_id, field_data in fields.items():
                    # Add FIELD node
                    if not G.has_node(field_id):
                        G.add_node(field_id, type="FIELD")
                    
                    # Link MODULE -> FIELD
                    G.add_edge(mod_id, field_id, rel="contains")
                    
                    injections = field_data.get("injections", {})
                    for pos, inj_id in injections.items():
                        # Add INJECTION node
                        if not G.has_node(inj_id):
                            G.add_node(inj_id, type="INJECTION")
                        
                        # Link FIELD -> INJECTION
                        G.add_edge(field_id, inj_id, rel="injection_at", pos=pos)
        
        return G


    def create_db(self):
        """
        collect all nodes values and stack with axis def
        -> uniform for all envs
        """
        self.db = self.get_empty_field_structure()

        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        for mid, m in modules:
            fields = self.g.get_neighbor_list(
                node=mid,
                target_type="FIELD",
            )

            if len(fields) > 0:
                #print(f"Fields for {mid} : {fields}")
                for fid, fattrs in fields.items():
                    self.db[m["module_index"]][fattrs["field_index"]] = [
                        fattrs["value"],
                        fattrs["axis_def"]
                    ]
            else:
                print(f"NO fields for {mid}: {fields}")
    

    def compile_pattern(self):
        try:
            compiler_struct = self.get_empty_field_structure()

            # GET MODULE todo filter for Guard
            modules = self.g.get_nodes(
                filter_key="type",
                filter_value="MODULE"
            )

            self.module_pattern_collector = []

            for i, module in enumerate(modules):
                mid = module["nid"]
                keys: list[str] = module["keys"]
                faxis_def: list[int or None] = module["axis_def"]


                # get module fields
                fields = self.g.get_neighbor_list(
                    node=mid,
                    target_type="FIELD",
                )

                meq = self.get_module_eqs(mid)

                # set empty param map struct struct
                modules_param_map = [
                    []
                    for _ in range(len(meq))
                ]

                for fid, fattrs in fields.items():
                    field_index = fattrs["field_index"]

                    for mindex, (eqid, eqattrs) in enumerate(meq):
                        # ALL PARAMS MAPPED OF METHOD
                        method_param_collector:list[
                            list[Any]
                        ] = []

                        axis_def = []

                        # get all params from
                        # todo mark order
                        params = self.g.get_neighbor_list(
                            node=eqid,
                            target_type="PARAM",
                        )

                        for pid, pattrs in params.items():
                            pval = pattrs["value"]

                            # SELF PARAM
                            if pid in keys:
                                pindex = keys.index(pid)

                                axis_def.append(faxis_def[pindex])
                                method_param_collector.append(
                                        [
                                            j,
                                            pindex,
                                            nfield_index,
                                        ]

                                )

                            # ENV ATTR
                            elif pid in self.env:

                                pindex = list(self.env.keys()).index(pid)
                                # add axis def
                                axis_def.append(None)

                                modules_param_map[i].append(
                                    [
                                        0, # envc altimes module 0
                                        pindex,
                                        0 # envc single field
                                    ]
                                )
                            else:
                                # NEIGHBOR VALUE
                                for j, module in enumerate(modules):
                                    nmkeys: list[str] = module["keys"]
                                    if pid in nmkeys:
                                        pindex = nmkeys.index(pid)

                                        # loop neighbors of field
                                        fneighbors = self.g.get_neighbor_list(
                                            node=fid,
                                            target_type="has_finteractant",
                                        )

                                        # get neighbors from field
                                        # mark: nfield-index represents row
                                        # of GT
                                        for nfid, nfattrs in fneighbors.items():
                                            nfield_index = nfattrs["field_index"]
                                            nfaxis_def: list[str] = module["axis_def"]

                                            # add axis def
                                            axis_def.append(
                                                nfaxis_def[nfield_index]
                                            )

                                            # append method to
                                            method_param_collector.append(
                                                [
                                                    j,
                                                    pindex,
                                                    nfield_index,
                                                ]
                                            )

                        return_index_map = [
                            i,
                            keys.index(eqattrs["return_key"]),
                            field_index,
                        ]

                        # save equation interaction pattern
                        modules_param_map[i].append(
                            [
                                eqattrs["callable"],
                                method_param_collector,
                                return_index_map,
                                axis_def,
                                mindex,
                            ]
                        )
                        """
                        GNNModuleBase(
                            runnable=,
                            inp_patterns=method_param_collector,
                            outp_pattern=return_index_map,
                            in_axes_def=axis_def,
                            method_id=mindex,
                        )"""

                        print(f"EQ pattern for {eqid} written")

                    self.module_pattern_collector.append(
                        modules_param_map
                    )
                    """
                    GNNChain(
                            method_modules=modules_param_map
                        )
                    """
                    print(f"EQ interaction pattern set up", self.module_pattern_collector)
        except Exception as e:
            print(f"Err compile_pattern: {e}")

    """
    compile self.module_pattern_collector -> GNNChain
    compile modules_param_map -> GNNModuleBase
    """

    def get_module_eqs(self, mid):
        # get methods for module
        meq = self.g.get_neighbor_list(
            node=mid,
            target_type="METHOD",
        )

        # bring to exec order
        meq = sorted(meq.values(), key=lambda x: x["method_index"], reverse=True)
        return meq

try:
    import ray
    from ray import get_actor
    @ray.remote
    class GuardWorker(Guard):
        def __init__(
            self,
            qfu,
            g,
            user_id
        ):
            Guard.__init__(
                self,
                qfu,
                g,
                user_id
            )
except Exception as e:
    print("Ray not accessible:", e)


"""


    def standard_manager(self, user_id: str = "public"):
        Upsert standard nodes and edges from QFUtils to BigQuery tables.
        print("PROCESSING STANDARD MANAGER WORKFLOW")
        standard_stack_exists: bool = self.user_manager.get_standard_stack(
            user_id
        )
        if standard_stack_exists is False:
            # Create module stack
            # todo one time create -> create copy from public row (so user can edit it -< session related)
            qf = QFUtils(
                G=nx.Graph()
            )
            qf.build_interacion_G()
            module_creator = ModuleCreator(
                G=qf.g.G,
                qfu=qf
            )
            module_creator.load_sm()

            modules = []
            fields = []
            # Upsert Nodes & Edges
            logging.info("Upserting standard nodes...")
            for nid, attrs in qf.g.G.nodes(data=True):
                ntype = attrs.get("type")

                if ntype == "MODULE":
                    # Get PARAM neighbors
                    params = qf.g.get_neighbor_list(
                        target_type="PARAM",
                        node=nid,
                        just_id=True,
                    )

                    # Upsert Module
                    module_data = {
                        "id": attrs["nid"],
                        "file_type": None,
                        "binary_data": None,
                        "code": attrs["code"],
                        "user_id": user_id,
                        "created_at": datetime.utcnow().isoformat(),
                        "params": params
                    }
                    modules.append(module_data)

                elif ntype == "FIELD":
                    # Upsert Field
                    field_data = {
                        "id": nid,
                        "params": attrs
                    }
                    fields.append(field_data)

            self.module_db_manager.set_module(
                modules, user_id
            )

            self.field_manager.set_field(
                fields, user_id
            )

            # Upsert Edges
            mfs = []
            ffs = []
            logging.info("Upserting standard edges...")
            for u, v, attrs in qf.g.G.edges(data=True):
                src_layer = attrs.get("src_layer")
                trgt_layer = attrs.get("trgt_layer")

                if src_layer == "MODULE" and trgt_layer == "FIELD":
                    # Module -> Field
                    data = {
                        "id": generate_numeric_id(),
                        "module_id": u,
                        "field_id": v,
                        "user_id": user_id
                    }
                    mfs.append(data)

                elif src_layer == "FIELD" and trgt_layer == "FIELD":
                    # Field -> Field
                    data = {
                        "id": generate_numeric_id(),
                        "field_id": u,
                        "interactant_field_id": v,
                        "user_id": user_id
                    }
                    ffs.append(data)

            # UPSERT
            self.field_manager.link_module_field(mfs)
            self.field_manager.link_field_field(ffs)
        print("FINISHED SM WORKFLOW AFTER SECONDs")


"""
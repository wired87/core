import json
import pprint

from itertools import product
from typing import Any

from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin
from core.env_manager import EnvManager

from core.fields_manager.fields_lib import FieldsManager

from core.injection_manager import InjectionManager
from core.module_manager.ws_modules_manager import ModuleWsManager
from core.qbrain_manager import QBrainTableManager
from core.user_manager import UserManager
from data import ENV

from gnn_master.pattern_store import GStore
from core.module_manager.mcreator import ModuleCreator
from qf_utils.all_subs import ALL_SUBS
from workflows.deploy_sim import DeploymentHandler

class PatternMaster:

    def __init__(self, g):
        self.g=g
        self.modules_struct=[]
        self.schema_pos=[]

    def get_positions(self, amount, dim):
        # Returns a list of tuples representing all N-dimensional coordinates
        # from 0 to amount-1
        return list(product(range(amount), repeat=dim))


class Guard(
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
        self.qb=QBrainTableManager()
        self.world_cfg=None
        self.artifact_admin = ArtifactAdmin()

        self.time = 0

        self.qfu:QFUtils = qfu

        self.g = g

        self.ready_map = {
            k: False
            for k in ALL_SUBS
        }

        self.testing = True

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


    def create_nodes(self, env_id, env_data):
        print("create_nodes...")
        # RESET G
        self.g.G = nx.Graph()

        # 1. Collect IDs
        module_ids = set()
        field_ids = set()
        inj_ids = set()

        # 1. CREATE GRAPH SCHEMA
        self.g.add_node({
                "nid": env_id,
                "type": "ENV",
            }
        )

        modules = env_data.get("modules", {})

        for i, (mod_id, mod_data) in enumerate(modules.items()):
            mod_id = mod_id.upper()
            module_ids.add(mod_id)

            mod = {
                    "nid": mod_id,
                    "type": "MODULE",
                    "module_index": i,
                }
            print("ADD MODULE", mod)
            self.g.add_node(mod)

            fields = mod_data.get("fields", {})

            for j, (field_id, field_data) in enumerate(fields.items()):
                field_id = field_id.upper()
                field_ids.add(field_id)

                self.g.add_node(
                    {
                        "nid": field_id,
                        "type": "FIELD",
                        "field_index": j,
                    }
                )

                injections = field_data.get("injections", {})
                for pos, inj_id in injections.items():
                    # add clean inj id
                    inj_ids.add(inj_id)

                    inj_id = f"{pos}__{inj_id}"
                    self.g.add_node(
                        {
                            "nid": inj_id,
                            "type": "INJECTION",
                        }
                    )
        print("create_nodes... done")
        self.g.print_status_G()
        return module_ids, field_ids, inj_ids


    def create_edges(
            self,
            env_id,
            env_data,
    ):
        # 1. CREATE GRAPH SCHEMA
        modules = env_data.get("modules", {})
        for i, (mod_id, mod_data) in enumerate(modules.items()):
            mod_id = mod_id.upper()

            # ENV -> MODULE
            self.g.add_edge(
                env_id,
                mod_id,
                attrs={
                    "rel": "has_module",
                    "src_layer": "ENV",
                    "trgt_layer": "MODULE",
                }
            )

            fields = mod_data.get("fields", {})

            for j, (field_id, field_data) in enumerate(fields.items()):
                field_id = field_id.upper()

                # MODULE -> FIELD
                self.g.add_edge(mod_id, field_id, attrs={
                    "rel": "has_field",
                    "src_layer": "MODULE",
                    "trgt_layer": "FIELD",
                })

                injections = field_data.get("injections", {})
                for pos, inj_id in injections.items():
                    inj_id = f"{pos}__{inj_id}"

                    # FIELD -> INJ
                    self.g.add_edge(
                        field_id,
                        inj_id,
                        attrs={
                            "rel": "has_injection",
                            "src_layer": "FIELD",
                            "trgt_layer": "INJECTION",
                        }
                    )
        self.g.print_status_G()
        print("create_edges... done")


    def check_create_ghost_mod(self, env_id):
        ghost_mod_id = f"GHOST_MODULE"
        print(f"Creating Ghost Module: {ghost_mod_id}")

        modules = self.g.get_nodes(filter_key="type", filter_value="MODULE", just_id=True)
        print("check_create_ghost_mod modules", modules)

        # Create Module Node
        self.g.add_node(dict(
            nid=ghost_mod_id,
            type="MODULE",
            module_index=len(modules),
            status="ghost",
        ))

        # Link Env -> Module
        self.g.add_edge(
            env_id,
            ghost_mod_id,
            attrs={
                "rel": "has_module",
                "src_layer": "ENV",
                "trgt_layer": "MODULE",
            }
        )

        # Link Env -> Module ->
        # TREAT AS FIELD FOR PARAM GATHERING
        self.g.add_edge(
            ghost_mod_id,
            env_id,
            attrs={
                "rel": "has_field",
                "src_layer": "MODULE",
                "trgt_layer": "FIELD",
            }
        )
        print("check_create_ghost_mod... done")


    def handle_env(self, env_id):
        print("handle_env...")

        keys = list(ENV.keys()) + ["i"]
        values = list(ENV.values()) + [1j]
        axis_def = (None for _ in range(len(keys)))

        # FETCH ENV
        res = self.qb.row_from_id(
            [env_id],
            table=self.env_manager.TABLE_ID
        )

        # GET DATA
        res = res[0]
        self.g.update_node(
            dict(
                nid=env_id,
                type="ENV",
                keys=keys,
                values=values,
                axis_def=axis_def,
                **res
            )
        )

        fields = self.g.get_neighbor_list_rel(
            node="GHOST_MODULE",
            trgt_rel="has_field",
        )

        print("ADD ENV FIELD")
        self.g.add_node(
            dict(
                nid=env_id,
                type="FIELD",
                keys=keys,
                values=values,
                axis_def=axis_def,
                field_index=len(fields),
                sub_type="ENV"
            )
        )

        print("handle_env... done")


    def handle_field_interactants(self, field_ids:list[str], env_id:str):
        """
        Include Ghost fields as last options for parameter collection
        """
        print("update_edges...")
        for fid in field_ids:
            missing_fields = []
            fattrs = self.g.get_node(fid)
            print(f"get finteractants for {fid}", fattrs)

            interactant_fields:list[str] or str = fattrs["interactant_fields"]
            print(f"{fid} interactant fields: {interactant_fields}, {type(interactant_fields)}")
            if isinstance(interactant_fields, str):
                interactant_fields = json.dumps(interactant_fields)

            for fi in interactant_fields:
                if not self.g.G.has_node(fi):
                    missing_fields.append(fi)

            if len(missing_fields) > 0:
                res = self.qb.row_from_id(
                    missing_fields,
                    table=self.env_manager.TABLE_ID)
                fetched_modules = {item["id"]: item for item in res}

                print("fetched_modules")

                 # ADD MISSING FIELD NODES
                for mfid, mfattrs in fetched_modules.items():
                    values = json.loads(mfattrs.get("values"))
                    keys = json.loads(mfattrs.get("keys"))
                    field_ids = self.g.get_nodes(filter_key="type", filter_value="FIELD", just_id=True)
                    print(f"ADD FINTERACTANT NODE {mfid}")
                    self.g.add_node(
                        dict(
                            nid=mfid,
                            type="FIELD",
                            keys=keys,
                            value=values,
                            field_index=len(field_ids),
                            **{
                                k: v
                                for k, v in mfattrs.items()
                                if k not in ["values", "keys"]
                            }
                        )
                    )

                    # MODULE -> FIELD
                    if "module_id" in mfattrs and mfattrs.get("module_id") is not None:
                        if self.g.G.has_node(mfattrs["module_id"]):
                            print("FIELD INTERACTANT -> MODULE")
                            self.g.add_edge(
                                fid,
                                mfid,
                                attrs={
                                    "rel": "has_field",
                                    "src_layer": "MODULE",
                                    "trgt_layer": "FIELD",
                                }
                            )
                        else:
                            print("MODULE WAS NOT CHOSSED BY USER -> GHOST FIELD")
                            self.check_create_ghost_mod(
                                env_id
                            )
                            self.g.add_edge(
                                fid,
                                mfid,
                                attrs={
                                    "rel": "has_finteractant",
                                    "src_layer": "FIELD",
                                    "trgt_layer": "FIELD",
                                }
                            )

                    # FIELD -> FI
                    self.g.add_edge(
                        fid,
                        mfid,
                        attrs={
                            "rel": "has_finteractant",
                            "src_layer": "FIELD",
                            "trgt_layer": "FIELD",
                        }
                    )

            # ADD FIELD FINTERACTANT
            for mfid in interactant_fields:
                self.g.add_edge(
                    fid,
                    mfid,
                    attrs={
                        "rel": "has_finteractant",
                        "src_layer": "FIELD",
                        "trgt_layer": "FIELD",
                    }
                )
        print("update_edges... done")
        #ghost_field_params


    def sim_start_process(self, env_id, env_data):
        """
        1. Parse payload to get IDs.
        2. Batch fetch data.
        3. Convert to pattern (Graph).
        4. Compile pattern.
        """
        print("sim_start_process...")

        # XTRCT NODE IDS
        module_ids, field_ids, inj_ids = self.create_nodes(
            env_id,
            env_data
        )
        print(f"Node IDs extracted: modules={len(module_ids)}, fields={len(field_ids)}, injections={len(inj_ids)}")

        self.create_edges(
            env_id,
            env_data
        )

        print("Edges created.")
        self.check_create_ghost_mod(env_id)

        self.handle_env(env_id)
        print("Env handled.")

        fetched_modules = {}
        if module_ids:
            print("Fetching modules from BQ...")
            res = self.qb.row_from_id(
                list(module_ids),
                table=self.module_db_manager.MODULES_TABLE
            )
            fetched_modules = {item["id"]: item for item in res}
            print(f"Fetched {len(fetched_modules)} modules.")

        # GET FIELDS BQ
        fetched_fields = {}
        if field_ids:
            print("Fetching fields from BQ...")
            res = self.qb.row_from_id(list(field_ids), table=self.field_manager.FIELDS_TABLE)
            fetched_fields = {item["id"]: item for item in res}
            print(f"Fetched {len(fetched_fields)} fields.")
            for fid, fattrs in fetched_fields.items():
                fid = fid.upper()
                if fattrs:
                    # Parse interactant_fields if string
                    if "interactant_fields" in fattrs and isinstance(fattrs["interactant_fields"], str):
                        try:
                            fattrs["interactant_fields"] = json.loads(fattrs["interactant_fields"])
                        except Exception as e:
                            print(f"Error parsing interactant_fields for {fid}: {e}")

                    self.g.update_node(dict(
                        nid=fid,
                        type="FIELD",
                        **fattrs
                    ))

                else:
                    self.g.update_node({"nid": fid, "type": "FIELD"})

        # GET INJ BQ
        fetched_injections = {}
        if inj_ids:
            print("Fetching injections from BQ...")
            res = self.qb.row_from_id(
                list(inj_ids),
                table=self.injection_manager.table
            )
            fetched_injections = {item["id"]: item for item in res}
            print(f"Fetched injections: {len(fetched_injections)} ")


        modules_config = env_data.get("modules", {})
        print(f"Processing {len(modules_config)} modules from config...")
        for mod_id, mod_config in modules_config.items():
            mod_id = mod_id.upper()
            print(f"  Processing module: {mod_id}")
            mod_data = fetched_modules.get(mod_id)
            if mod_data:
                self.g.update_node(dict(
                    nid=mod_id,
                    type="MODULE",
                    **{k:v for k,v in mod_data.items() if k not in self.g.get_node(nid=mod_id)}
                ))

            else:
                self.g.update_node({"nid": mod_id, "type": "MODULE"})

            fields_config = mod_config.get("fields", {})
            print(f"Processing {len(fields_config)} fields for module {mod_id}...")

            for field_id, field_config in fields_config.items():
                field_id = field_id.upper()
                print(f"Processing field: {field_id}")
                field_data = fetched_fields.get(field_id)
                if field_data:

                    values = json.loads(field_data.get("values"))
                    keys = json.loads(field_data.get("keys"))
                    
                    if "interactant_fields" in field_data and isinstance(field_data["interactant_fields"], str):
                        try:
                            field_data["interactant_fields"] = json.loads(field_data["interactant_fields"])
                        except Exception as e:
                            print(f"Error parsing field data interactant_fields for {field_id}: {e}")

                    self.g.update_node(
                        dict(
                            nid=field_id,
                            type="FIELD",
                            keys=keys,
                            value=values,
                            **{
                                k:v
                                for k,v in field_data.items()
                                if k not in ["values", "keys"]
                            }
                        )
                    )

                    try:
                        self.qfu.add_params_link_fields(
                            keys,
                            values,
                            field_id,
                            mod_id
                        )
                    except Exception as e:
                        print(f"Error linking params for field {field_id}: {e}")
                else:
                    print(f"no field data for field {field_id}")
                    self.g.update_node({
                        "nid": field_id,
                        "type": "FIELD",
                    })

                injections_config = field_config.get("injections", {})
                print(f"Processing {len(injections_config)} injections for field {field_id}...")
                for pos, inj_id in injections_config.items():
                    inj_data = fetched_injections.get(inj_id)
                    inj_id = f"{pos}__{inj_id}"
                    print(f"inj_data: {inj_data}")
                    if inj_data:
                        self.g.add_node(
                            dict(
                                nid=inj_id,
                                type="INJECTION",
                                **inj_data
                            )
                        )

                    else:
                        self.g.update_node({"nid": inj_id, "type": "INJECTION", })

        print("Handling field interactants...")
        self.handle_field_interactants(
            list(field_ids),
            env_id
        )

        try:
            print("Calling main...")
            self.main(
                env_id,
                env_node=self.g.get_node(env_id)
            )
            print("Pattern Compiled:")
        except Exception as e:
            print(f"Error in Errormain process: {e}")
        print("Graph populated with fetched data.")




    def main(self, env_id:str, env_node:dict):
        """
        CREATE/COLL ECT PATTERNS FOR ALL ENVS AND CREATE VM
        """
        print("Main started...")

        compiled_patterns: list = self.compile_pattern(env_id)


        dbs = self.create_db(env_id)

        injection_patterns = self.set_inj_pattern(env_node)

        vm_payload = self.create_vm_cfgs(
            env_id=env_id,
            compiled_patterns=compiled_patterns,
            dbs=dbs,
            injection_patterns=injection_patterns,
        )
        #self.deploy_vms(vm_payload)
        print("Main... done")


    def deploy_vms(self, vm_payload):
        print("Deploying VMs...")
        for cfg in vm_payload.values():
            self.deployment_handler.create_vm(
                cfg=cfg
            )
        print(f"Deployment Finished!")


    def create_vm_cfgs(
            self,
            env_id: str,
            compiled_patterns: list,
            dbs: Any,
            injection_patterns: list,
        ):
        vm_payload = {}
        env_node = self.g.get_node(env_id)
        env_node = {k:v for k,v in env_node.items() if k not in ["updated_at", "created_at"]}
        #print("add env node to world cfg:", env_node)

        # CREATE VM PAYLOAD
        world_cfg = {
            "UPDATOR_PATTERN": compiled_patterns,
            "DB": dbs,
            "AMOUNT_NODES": env_node.get("amount_of_nodes", 100),
            "INJECTION_PATTERN": injection_patterns,
            "WORLD_CFG": json.dumps({
                "sim_time": env_node.get("sim_time", 1),
                "amount_nodes": env_node.get("amount_of_nodes", 1),
                "dims": env_node.get("dims", 3),
            }),
            #"ENERGY_MAP": param_pathway
        }

        # SAVE to DB
        self.update_env_pattern(env_id=env_id, pattern=world_cfg)

        # Etend env var cfg structure
        container_env = self.deployment_handler.env_creator.create_env_variables(
            env_id=env_id,
            cfg=world_cfg
        )

        # get cfg
        vm_payload[env_id] = self.deployment_handler.get_vm_cfg(
            instance_name=env_id,
            testing=self.testing,
            image=self.artifact_admin.get_latest_image(),
            container_env=container_env
        )
        print(vm_payload)
        print("create_vm_cfgs finished... done")
        return vm_payload


    def update_env_pattern(self, env_id, pattern):
        try:
            self.module_db_manager.qb.update_env_pattern(
                env_id=env_id,
                pattern_data=pattern,
                user_id=self.user_id
            )
        except Exception as e:
            print(f"Error updating env pattern: {e}")

    def get_state(self):
        return len(self.fields) and self.world_cfg




    def set_wcfg(self, world_cfg):
        self.world_cfg = world_cfg

        amount_nodes = world_cfg["amount_of_nodes"]

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


    def set_inj_pattern(
            self,
            env_attrs,
            trgt_keys=["energy", "j_nu", "vev"],# injectable parameters
    ) -> list[list[int]]:
        # exact same format
        print("set_inj_pattern...")
        struct = []
        try:
            amount_nodes = env_attrs["amount_of_nodes"]
            dims = env_attrs["dims"]
            schema_positions = self.get_positions(amount_nodes, dims)

            modules = self.g.get_nodes(
                filter_key="type",
                filter_value="MODULE"
            )

            for i, (mid, mattrs) in enumerate(modules):
                if "GHOST" in mid.upper(): continue

                # get param (method) neighbors
                # get params
                # extract index
                if "module_index" not in mattrs:
                    continue

                midx = mattrs["module_index"]
                print(f"MODULE index for {mid}: {midx}")

                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True
                )

                for fid, fattrs in fields.items():
                    if "field_index" not in fattrs:
                        continue

                    fi:int = fattrs["field_index"]
                    f_keys=fattrs.get("keys", [])


                    trgt_index = None
                    for key_opt in trgt_keys:
                        if key_opt in f_keys:
                            # CKPT PARAM INDEX
                            param_trgt_index = f_keys.index(key_opt)
                            print(f"{key_opt} in {fid}s params:", param_trgt_index)

                            # 5. Loop Injections
                            injections = self.g.get_neighbor_list_rel(
                                node=fid,
                                trgt_rel="has_injection",
                            )

                            if len(injections) > 0:
                                print(f"INJs for FIELD {fid}:", injections)

                            if not injections or not len(injections):
                                print(f"no injections for field {fid}")
                                continue

                            for inj_id, inj_attrs in injections:
                                print("work inj_id:", inj_id)
                                pos_index:int or None = schema_positions.index(tuple(eval(inj_id.split("__")[0])))

                                print(f"pos_index {inj_id}:", pos_index)
                                struct.append(
                                    (midx, fi, param_trgt_index, pos_index, inj_attrs["data"])
                                )
                                print("applied data to struct")
                            break
                    print(f"set param pathway db from mod {midx} -> field {fattrs['nid']}({fi}) = {trgt_index}")
            print(f"set_inj_pattern... done")
        except Exception as e:
            print("Err set_inj_pattern", e)
        return struct








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


    def create_db(self, env_id):
        """
        collect all nodes values and stack with axis def
        -> uniform for all envs
        """
        db = self.get_empty_field_structure()
        modules = self.g.get_neighbor_list(
            node=env_id,
            target_type="MODULE",
        )

        for mid, m in modules.items():
            # get module specific fields
            if "GHOST" in mid.upper(): continue
            fields = self.g.get_neighbor_list_rel(
                node=mid,
                trgt_rel="has_field",
                as_dict=True,
            )
            mod_idx = m["module_index"]
            print("CREATE DB FIELDS FETCHED:", fields)

            for fid, fattrs in fields.items():
                try:
                    if fattrs.get("field_index") is None:
                         print(f"MISSING OR NONE FIELD INDEX FOR {fid}: keys: {fattrs.keys()}")
                         continue
                    fi = fattrs["field_index"]
                    print(f"get db from mod{mod_idx} -> field {fi}")
                    db[mod_idx][fi] = [
                        fattrs.get("value", []),
                        fattrs.get("axis_def", [])
                    ]
                except Exception as e:
                    print(f"Err create_db for interactant {fattrs}:", e)

        print("create_db finished successfully")
        return db


    def compile_pattern(self, env_id):
        print("compile_pattern...")
        env_struct = self.get_empty_field_structure()
        
        try:
            modules = self.g.get_neighbor_list_rel(
                node=env_id,
                trgt_rel="has_module",
                as_dict=True,
            )

            print(f"modules for {env_id}", modules)

            # ADD ENV AND GHOST MODULE DIM
            env_struct.append([])

            for mid, module in modules.items():
                if "GHOST" in mid.upper(): continue

                m_idx = module.get("module_index")

                if m_idx is None:
                    print(f"Skipping module {mid} without index")
                    continue

                print(f"compiler work mod {mid} at index {m_idx}")

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                meq:list[dict] = json.loads(module["arsenal_struct"])
                print("meq", meq, type(meq))

                modules_param_map = [
                    []
                    for _ in range(len(meq))
                ]

                if len(list(fields.keys())) == 0: continue

                # Iterate Fields
                for fid, fattrs in fields.items():
                    print(f"  Field: {fid}:{fattrs}")
                    field_index = fattrs.get("field_index")
                    print("field index:", field_index)

                    keys: list[str] = fattrs.get("keys", [])
                    print("keys:", keys)

                    if field_index is None: continue

                    # Iterate Equations
                    for eq_idx, eqattrs in enumerate(meq):
                        print("eqattrs", eqattrs)
                        eqid = eqattrs.get("nid")
                        print(f"    Equation: {eqid}")

                        # ALL PARAMS MAPPED OF METHOD
                        method_param_collector = []
                        axis_def = []

                        # get all params from METHOD
                        params = eqattrs.get("params", [])
                        print(f"      Params: {params}")

                        # LOOP EQ-PARAMS
                        for pid in params:
                            print("work pid", pid)

                            # SELF PARAM (Field's own module)
                            if pid in keys:
                                print(f"{pid} in {fid}")
                                pindex = keys.index(pid)

                                method_param_collector.append(
                                    [
                                        m_idx,
                                        pindex,
                                        field_index,
                                    ]
                                )
                                print(f"        Mapped Self Param: {pid} -> {[m_idx, pindex, field_index]}")

                                # ensure single attrs
                                if eq_idx == 0:
                                    axis_def.append(fattrs["axis_def"][pindex])

                            # param key in any diferent modules interactant field?
                            else:
                                print("Key in interactant")
                                # loop neighbors of field to find specific targets
                                fneighbors = self.g.get_neighbor_list_rel(
                                    node=fid,
                                    trgt_rel="has_finteractant",
                                    as_dict=True
                                )
                                #print("fneighbors", fneighbors)
                                if len(list(fneighbors.keys())) == 0: continue

                                for o, (finid, fiattrs) in enumerate(fneighbors.items()):
                                    print("interactant fnid", finid, o)
                                    ikeys = fiattrs.get("keys", [])
                                    #print("ikeys",ikeys)
                                    if pid in ikeys:
                                        print("interactant pid", pid, o)

                                        nfield_index = fiattrs.get("field_index")
                                        print("interactant nfield_index", nfield_index, o)

                                        pindex = fiattrs["params"].index(pid)
                                        print("interactant pindex", pindex, o)

                                        # Get neighbor field's module to get its index
                                        pmod_id = fiattrs["module_id"]
                                        print("interactant pmod_id", pmod_id, o)

                                        pmod = self.g.get_node(nid=pmod_id)
                                        print("interactant pmod", pmod, o)

                                        mod_index = pmod.get("module_index")
                                        print("interactant mod_index", mod_index, o)

                                        method_param_collector.append(
                                            [
                                                mod_index,
                                                pindex,
                                                nfield_index,
                                            ]
                                        )
                                        print(f"        Mapped Neighbor Param: {pid} -> {[mod_index, pindex, nfield_index]}")
                                        # ensure single axis def for method set
                                        if eq_idx == 0 and o == 0:
                                            axis_def.append(fiattrs["axis_def"][pindex])

                        # Return Map
                        return_key = eqattrs.get("return_key")
                        print("return_key", return_key)
                        ret_idx = keys.index(
                            return_key
                        ) if return_key in keys else 0
                        #print("ret_idx", ret_idx)

                        return_index_map = [
                            m_idx,
                            ret_idx,
                            field_index,
                        ]
                        #print(f"      Return Map: {return_key} -> {return_index_map}")

                        # save equation interaction pattern
                        # Use eq_idx (Equation Index)
                        pattern_entry = [
                                eqattrs.get("docstring"),
                                method_param_collector,
                                return_index_map,
                                axis_def, # collection for methods axis def
                                eq_idx,
                            ]
                        modules_param_map[eq_idx].append(pattern_entry)
                        print(f"      Pattern Entry Added: {pattern_entry}")

                        #print(f"EQ pattern for {eqid} written")

                # DONE WITH MODULE'S FIELDS
                # Append this module's pattern to env_struct
                print(f"Saving compiled pattern to env_struct[{m_idx}]")
                env_struct[m_idx] = modules_param_map

            #pprint.pp(env_struct)
            print("compile_pattern... done")
        except Exception as e:
            print(f"Err compile_pattern: {e}")
        return env_struct




    def create_actor(self):
        try:
            import ray
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


    def get_module_eqs(self, mid):
        # get methods for module
        meq = self.g.get_neighbor_list(
            node=mid,
            target_type="METHOD",
        )

        # bring to exec order
        meq = sorted(meq.values(), key=lambda x: x["method_index"], reverse=True)
        return meq


    def get_empty_field_structure(self, include_ghost_mod=True, set_none=False):
        modules: list = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )
        try:
            # 1. Calculate Max Module Index
            for mid, m in modules:
                print(f">>>module {mid}", m["module_index"])

            #nit with size max_index + 1
            modules_struct = [
                [] for _ in range(len(modules))
            ]
            print("modules_struct initialized size:", len(modules_struct))

            for i, (mid, m) in enumerate(modules):
                if not include_ghost_mod:
                    if "GHOST" in mid.upper(): continue

                m_idx = m.get("module_index")
                print(f"m_idx for {mid}:{m_idx}")
                if m_idx is None:
                    continue

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                print(f"get_empty_field_structure fields for {mid}")

                field_struct = [
                    None if set_none is False else []
                    for _ in range(len(fields))
                ]
                print(f"field_struct created for {mid}")
                # SET EMPTY FIELDS STRUCT AT MODULE INDEX
                modules_struct[m_idx] = field_struct

            pprint.pp(modules_struct)
            print("modules_struct... done")
            return modules_struct
        except Exception as e:
            print("Err get_empty_field struct:", e)



if __name__ == "__main__":
    import networkx as nx
    from utils.graph.local_graph_utils import GUtils
    from qf_utils.qf_utils import QFUtils

    # Define payload
    payload = {'type': 'START_SIM', 'data': {'config': {'env_7c87bb26138a427eb93cab27d0f5429f': {'modules': {'GAUGE': {'fields': {'photon': {'injections': {'[4,4,4]': 'hi'}}, 'w_plus': {'injections': {}}, 'w_minus': {'injections': {}}, 'z_boson': {'injections': {}}, 'gluon_0': {'injections': {}}, 'gluon_1': {'injections': {}}, 'gluon_2': {'injections': {}}, 'gluon_3': {'injections': {}}, 'gluon_4': {'injections': {}}, 'gluon_5': {'injections': {}}, 'gluon_6': {'injections': {}}, 'gluon_7': {'injections': {}}}}, 'HIGGS': {'fields': {'phi': {'injections': {}}}}, 'FERMION': {'fields': {'electron': {'injections': {}}, 'muon': {'injections': {}}, 'tau': {'injections': {}}, 'electron_neutrino': {'injections': {}}, 'muon_neutrino': {'injections': {}}, 'tau_neutrino': {'injections': {}}, 'up_quark_0': {'injections': {}}, 'up_quark_1': {'injections': {}}, 'up_quark_2': {'injections': {}}, 'down_quark_0': {'injections': {}}, 'down_quark_1': {'injections': {}}, 'down_quark_2': {'injections': {}}, 'charm_quark_0': {'injections': {}}, 'charm_quark_1': {'injections': {}}, 'charm_quark_2': {'injections': {}}, 'strange_quark_0': {'injections': {}}, 'strange_quark_1': {'injections': {}}, 'strange_quark_2': {'injections': {}}, 'top_quark_0': {'injections': {}}, 'top_quark_1': {'injections': {}}, 'top_quark_2': {'injections': {}}, 'bottom_quark_0': {'injections': {}}, 'bottom_quark_1': {'injections': {}}, 'bottom_quark_2': {'injections': {}}}}}}}}, 'auth': {'session_id': 339617269692277, 'user_id': '72b74d5214564004a3a86f441a4a112f'}, 'timestamp': '2026-01-08T11:54:50.417Z'}

    print("Running START_SIM test...")
    
    g = GUtils(G=nx.Graph())
    qfu = QFUtils(g=g)
    user_id = payload['auth']['user_id']
    
    guard = Guard(qfu, g, user_id)

    guard.sim_start_process(
        env_id="env_7c87bb26138a427eb93cab27d0f5429f",
        env_data=payload["data"]["config"]["env_7c87bb26138a427eb93cab27d0f5429f"],
    )
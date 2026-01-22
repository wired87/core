import json
import pprint

from itertools import product
from typing import Any

from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin
from core.env_manager import EnvManager

from core.fields_manager.fields_lib import FieldsManager
from core.method_manager.method_lib import MethodManager

from core.injection_manager import InjectionManager
from core.module_manager.ws_modules_manager import ModuleWsManager
from core.param_manager.params_lib import ParamsManager
from core.qbrain_manager import QBrainTableManager
from core.user_manager import UserManager
from data import ENV

from core.module_manager.mcreator import ModuleCreator
from qf_utils.all_subs import ALL_SUBS
from utils.str_size import get_str_size
from utils.xtract_trailing_numbers import extract_trailing_numbers
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
        self.method_manager = MethodManager()
        self.injection_manager = InjectionManager()
        self.env_manager = EnvManager()
        self.user_manager = UserManager()
        self.param_manager = ParamsManager()
        self.qb=QBrainTableManager()
        print("DEBUG: QBrainTableManager initialized")
        self.world_cfg=None
        self.artifact_admin = ArtifactAdmin()
        print("DEBUG: ArtifactAdmin initialized")

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
        print("DEBUG: ModuleCreator initialized")

        #self.pattern_arsenal = GStore

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

        keys = list(self.qfu.create_env().keys())

        env_constants = self.qb.row_from_id(keys, table="params")
        values = [v["value"] for v in env_constants]
        axis_def = [v.get("axis_def", None) for v in env_constants]


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
                **res,
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



    def handle_methods(self, module_ids:list[str]):
        print("handle_methods...")
        for mid in module_ids:
            mod_node = self.g.get_node(nid=mid)
            mmethods = mod_node["methods"]

            if isinstance(mmethods, str):
                mmethods = json.loads(mmethods)

            print("mod_node methods", mmethods)

            methods = self.qb.row_from_id(
                nid=mmethods,
                table=self.method_manager.METHODS_TABLE,
            )
            print("methods", methods)

            methods = {
                item["id"]: item
                for item in methods
            }
            print("methods1", methods)

            for k, (method_id, method_data) in enumerate(methods.items()):
                self.g.add_node({
                    "nid": method_id,
                    "type": "METHOD",
                    **method_data,
                })

                # MODULE -> METHOD
                self.g.add_edge(
                    mid,
                    method_id,
                    attrs={
                        "rel": "has_method",
                        "src_layer": "MODULE",
                        "trgt_layer": "METHOD",
                    }
                )
        print("handle_methods... done")




    def handle_fields(self, module_ids:list[str]):
        for mid in module_ids:
            mod_node = self.g.get_node(nid=mid)
            fields = self.qb.row_from_id(
                list(mod_node["fields"]),
                table=self.field_manager.FIELDS_TABLE
            )
            fields = {
                item["id"]: item
                for item in fields
            }
            for k, (field_id, field_data) in enumerate(fields.items()):
                # MODULE -> FIELD
                self.g.add_edge(
                    mid,
                    field_id,
                    attrs={
                        "rel": "has_field",
                        "src_layer": "MODULE",
                        "trgt_layer": "FIELD",
                    }
                )



    def handle_field_interactants(self, field_ids:list[str], env_id:str):
        """
        Include Ghost fields as last options for parameter collection
        """
        print("update_edges...")
        for fid in field_ids:
            missing_fields = []
            fattrs = self.g.get_node(fid)
            #print(f"get finteractants for {fid}", fattrs)

            interactant_fields:list[str] or str = fattrs["interactant_fields"]
            print(f"{fid} interactant fields: {len(interactant_fields)}")
            if isinstance(interactant_fields, str):
                interactant_fields = json.loads(interactant_fields)

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

        self.handle_methods(module_ids=list(module_ids))


        print("Handling field interactants...")
        self.handle_field_interactants(
            list(field_ids),
            env_id
        )

        # ETEND FIELDS PARAMS WITH RETURN KEYS
        self.extend_fields_keys()

        try:
            print("Calling main...")
            self.main(
                env_id,
                env_node=self.g.get_node(env_id)
            )
            print("Pattern Compiled:")
        except Exception as e:
            print(f"Error in main process: {e}")
        print("Graph populated with fetched data.")


    def add_field_index(self, fid):
        """
        Case we have multiple field parts (e.g. quarks listb_quark_1) -> equation requires index -> extract hiere field part index and save within the field struct
        :param fid:
        :return:
        """
        numbers = extract_trailing_numbers(fid)
        if len(numbers):
            if "quark" in fid.lower():
                key = "quark_index"
            else:
                key = f"{fid.split(f'_{numbers}')[0]}_index".lower()
            return key, int(numbers)


    def main(self, env_id:str, env_node:dict):
        """
        CREATE/COLL ECT PATTERNS FOR ALL ENVS AND CREATE VM
        """
        print("Main started...")

        iterator_idx_map, DB, AXIS = self.create_db()

        transformation_kernels: dict = self.compile_pattern(
            DB,
            iterator_idx_map,
        )



        injection_patterns = self.set_inj_pattern(env_node)

        vm_payload = self.create_vm_cfgs(
            env_id=env_id,
            injection_patterns=injection_patterns,
            dbs=DB, # flatened array can bes accessed by fields-, and eqidx, eq_struct
            method_layer = self.method_layer(),
            transformation_kernels=transformation_kernels,
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
            dbs: Any,
            injection_patterns: list,
            method_layer: list,
            transformation_kernels:dict
        ):
        env_node = self.g.get_node(env_id)
        env_node = {k:v for k,v in env_node.items() if k not in ["updated_at", "created_at"]}
        #print("add env node to world cfg:", env_node)

        # BOB BUILDER ACTION
        world_cfg = {
            "DB": dbs,
            "INJECTION_PATTERN": injection_patterns,
            "METHOD_LAYER": method_layer,
            **transformation_kernels,

            "WORLD_CFG": {
                "sim_time": env_node.get("sim_time", 1),
                "amount_nodes": env_node.get("amount_of_nodes", 1),
                "dims": env_node.get("dims", 3),
            },
            #"ENERGY_MAP": param_pathway
        }
        cfg_str = json.dumps(world_cfg)

        #cfg_bytes = bytes(cfg_str.encode("utf-8"))

        if self.testing:
            with open(rf"C:\Users\bestb\PycharmProjects\BestBrain\test_out.json", "w", encoding="utf-8") as f:
                f.write(cfg_str)

        get_str_size(cfg_str)

        # SAVE to DB
        self.update_env_pattern(env_id=env_id, pattern=world_cfg)

        # Etend env var cfg structure
        container_env = self.deployment_handler.env_creator.create_env_variables(
            env_id=env_id,
            cfg=world_cfg
        )

        # get cfg
        vm_payload = self.deployment_handler.get_vm_cfg(
            instance_name=env_id,
            testing=self.testing,
            image=self.artifact_admin.get_latest_image(),
            container_env=container_env
        )
        #print(vm_payload)
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
            trgt_keys=["energy", "j_nu", "vev"], # injectable parameters
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

                    for key_opt in trgt_keys:
                        if key_opt in f_keys:
                            # CKPT PARAM INDEX
                            param_trgt_index = f_keys.index(key_opt)
                            #print(f"{key_opt} in {fid}s params:", param_trgt_index)

                            # 5. Loop Injections
                            injections = self.g.get_neighbor_list_rel(
                                node=fid,
                                trgt_rel="has_injection",
                            )

                            if len(injections) > 0:
                                print(f"INJs for FIELD {fid}:", len(injections))

                            if not injections or not len(injections):
                                #print(f"no injections for field {fid}")
                                continue

                            for inj_id, inj_attrs in injections:
                                print("work inj_id:", inj_id)
                                pos_index:int or None = schema_positions.index(tuple(eval(inj_id.split("__")[0])))

                                print(f"pos_index {inj_id}:", pos_index)
                                inj_pathway = (midx, fi, param_trgt_index, pos_index, inj_attrs["data"])
                                struct.append(inj_pathway)
                                print(
                                    f"set param pathway db from mod {midx} -> field {fattrs['nid']}({fi}) = {inj_pathway}"
                                )
                            break
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


    def create_db(self):
        """
        collect all nodes values and stack with axis def
        -> uniform for all envs
        """
        print("create_db...")

        db = self.get_empty_field_structure()
        axis = self.get_empty_field_structure()

        # 1d db save all params values
        # todo: need first sepparate to nested struct (because index must be persistent)
        DB = []
        AXIS = []

        try:

            modules: list = self.g.get_nodes(
                filter_key="type",
                filter_value="MODULE",
            )

            for mid, m in modules:
                m_idx = m.get("module_index")

                print("create_db work mod", mid, m_idx)
                # get module specific fields
                if "GHOST" in mid.upper(): continue

                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True
                )

                # print(f"create_db fields for {mid}: ", fields)

                for fid, fattrs in fields.items():
                    fidx = fattrs.get("field_index")

                    print("apply db pattern to", m_idx, fidx)

                    xdef = fattrs.get("axis_def", [])
                    if isinstance(xdef, str):
                        xdef = json.loads(fattrs.get("axis_def", []))

                    vals = fattrs.get("values", [])
                    if isinstance(xdef, str):
                        vals = json.loads(fattrs.get("values", []))
                    
                    db[m_idx][fidx] = vals
                    axis[m_idx][fidx] = xdef

            iterator_idx_map = [len(o) for item in db for o in item]
            DB.extend(o for item in db for o in item)
            AXIS.extend(o for item in db for o in item)

        except Exception as e:
            print(f"Err create_db:", e)
        print("create_db... done")
        return iterator_idx_map, DB, AXIS


    def method_layer(self):
        print("method_layer... ")
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        mod_len_exclude_ghost = len(modules)-1
        print("method_layer mod_len_exclude_ghost", mod_len_exclude_ghost)

        # ADD ENV AND GHOST MODULE DIM
        method_struct = self.get_empty_field_structure()

        try:
            print("method_layer compilation...")
            for mid, module in modules:
                # ghost does not have equation
                if "GHOST" in mid.upper(): continue
                print("method_layer... working", mid)

                midx = module.get("module_index")
                print("midx", midx)

                mod_meth_struct = []
                print("method_layer... module_index")

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                mlen  = len(list(methods.keys()))
                if not mlen:
                    print("method_layer... len methods", mlen)
                    continue

                print("method_layer eqs", type(methods))

                # Iterate Equations
                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    # print("eqattrs", eqattrs)
                    #print(f"Equation: {eqid}")

                    # PARAMS from METHOD
                    params = eqattrs.get("params", [])
                    if isinstance(params, str):
                        params = json.loads(params)
                    print(f"Params: {len(params)}")

                    # ENTIRE METHOD SPECIFIC PARAMS
                    method_struct_item = [
                        eqattrs.get("jax_code", eqattrs.get("code")),
                        eqattrs.get("axis_def"),
                    ]
                    mod_meth_struct.append(method_struct_item)
                method_struct.append(mod_meth_struct)

        except Exception as e:
            print("Err method_layer", e)
        #pprint.pp(method_struct)
        print("method_layer... done")
        return method_struct



    def compile_pattern(self, db, iterator_db_idx_map):
        print("compile_pattern...")
        # todo for field index includ again

        # db out gnn
        #db_out_gnn = self.get_empty_field_structure(include_ghost_mod=False)

        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        mod_len_exclude_ghost = len(modules)-1
        print("mod_len_exclude_ghost", mod_len_exclude_ghost)

        print("get shost mods...")
        ghost_mod = self.g.get_node(nid="GHOST_MODULE")
        print("ghost_mod:", ghost_mod)

        gmod_idx = ghost_mod.get("module_index")
        print("gmod_idx:", gmod_idx)

        ghost_fields = self.g.get_neighbor_list_rel(
            trgt_rel="has_field",
            node="GHOST_MODULE",
        )

        print("ghost_fields:", len(ghost_fields))


        iterator = {
            # GOAL FOR THE ITERATOR COMPONENT
            # map int to feature
            # map feature -> field (&field -> mod)
            # e.g. given index 1000 : len features * sum(features = amount all features // (sum(fields) / len(fields)) = field index

            # index space for equation len per module # save len eq for each module
            "modules": [], # 5, 6, 8 = 3 modules with n eqs

            # amount params for each method
            "method_param": [], # e.g. 3*5, 7params to 6 eqs, ...

            # index space len fields per module (why module? because each module and underlying equation uses same fields (in this model)
            "fields": [], # 19, 5, 21 # each module/eq corresponds to n fields

            # param index map
            "db_params": iterator_db_idx_map, # 19*20, (2*100, 3*10), ... the amount of params each field has

            # amount features for each field for each eq (must be for field since each field has potential different amount interactants)
            "field_variations": [], # (5 eqs * 19 fields) * 30 variations,

            # get vaiation indices


            #


            #

        }

        # collection for hard data 1d scaled
        hardware = {
            "param_axis_def": [], # use also for method axis (based on value)
            "method_struct": [],
            "db": db,
            "db_def_idx_map": [], # edge def -> db
            "return_key_map": [],
        }

        try:
            print("start compilation...")
            for m_idx, (mid, module) in enumerate(modules):
                print("compile_pattern... working", mid)
                if "GHOST" in mid.upper(): continue

                print("compile_pattern... module_index", m_idx)

                if m_idx is None:
                    print(f"Skipping module {mid} without index")
                    continue

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                if not len(list(methods.keys())):
                    print("compile_pattern... len methods 0")
                    continue

                # Lets append the len of the methods to the code struct
                iterator["modules"].append(len(list(methods.keys())))

                print("compile_pattern meq", type(methods))

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                len_fields = len(list(fields.keys()))
                print("fields:", len_fields)
                if len(list(fields.keys())) == 0: raise Exception("Err: no fields found...")
                iterator["fields"].append(len_fields)

                # MARK: wenn wir gleichungen als äußerstes element haben dann erstellen wir automatische einen bidirketionalen
                # Graphen (eq->field->param_struct zeigt auf struct mit param mappings zu midx->field->param)

                # featurs dürfen nicht gemerget werden -> da direkte signal Verfolgung sons verloren geht (unter geht)
                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    # print("eqattrs", eqattrs)
                    print(f"Equation: {eqid}")

                    # field param blocks collector
                    eq_param_collector = []

                    # PARAMS from METHOD
                    params = eqattrs.get("params", [])
                    print("methods params", params)

                    # PARAMS from METHOD
                    return_key = eqattrs.get("return_key", [])
                    print(f"methods {eqid} return_key", return_key)

                    # params_origin
                    params_origin = eqattrs.get("origin", None)

                    if params_origin is None:
                        params_origin = ["" for _ in range(len(params))]
                    print("methods params_origin", params_origin)

                    if isinstance(params, str):
                        params = json.loads(params)
                    print(f"Params: {len(params)}")

                    for field_index, (fid, fattrs) in enumerate(fields.items()):
                        #print(f"Field: {fid}", field_index)

                        fidx = module.get("field_index")
                        print("fidx", fidx)

                        # Space to save all variations for all inteactions for all equations
                        field_eq_param_struct = []

                        if isinstance(fattrs["keys"], str):
                            fattrs["keys"] = json.loads(fattrs["keys"])

                        if isinstance(fattrs["values"], str):
                            fattrs["values"] = json.loads(fattrs["values"])

                        keys: list[str] or str = fattrs.get("keys", [])

                        if isinstance(keys, str):
                            keys = json.loads(keys)
                        print(f"{fid} keys:", keys)

                        fneighbors = self.g.get_neighbor_list_rel(
                            node=fid,
                            trgt_rel="has_finteractant",
                            as_dict=True
                        )

                        # LOOP EQ-PARAMS
                        for pidx, pid in enumerate(params):
                            collected = False

                            param_collector = []
                            print("work pid", pid)

                            # SELF PARAM (Field's own module)
                            is_prefixed = pid.endswith("_")

                            EXCLUDED_ORIGINS = ["neighbor", "interactant"]

                            if (
                                pid in keys and not is_prefixed and params_origin[pidx] not in EXCLUDED_ORIGINS
                            ):
                                print(f"{pid} in {fid}")

                                pindex = keys.index(pid)

                                field_eq_param_struct.append(
                                    self.get_db_index(
                                        db,
                                        m_idx,
                                        field_index,
                                        pindex,
                                    )
                                )
                                collected=True
                                print(f"Mapped Self Param: {pid} -> {fid}")

                            else:
                                print(f"param {pid} not in {keys}")

                                for _ in range(len(pid)):
                                    if is_prefixed:
                                        print("remove slicing end char from", pid)
                                        pid = pid[:-1]
                                        print("edited pid", pid)
                                        break
                                    else:
                                        break

                                for o, (finid, fiattrs) in enumerate(fneighbors.items()):
                                    print("check interactant fnid", finid, o)
                                    ikeys = fiattrs.get("keys")

                                    if isinstance(ikeys, str):
                                        print("convert ikeys", ikeys)
                                        ikeys = json.loads(ikeys)

                                    # param key in interactant field?
                                    if pid in ikeys:
                                        # collect maps for all interactants
                                        #print("interactant pid", pid, o)

                                        nfield_index = fiattrs.get("field_index")
                                        #print("interactant nfield_index", nfield_index, o)

                                        pindex = ikeys.index(pid)
                                        #print("interactant pindex", pindex, o)

                                        # Get neighbor field's module to get its index
                                        pmod_id = fiattrs["module_id"]
                                        #print("interactant pmod_id", pmod_id, o)

                                        pmod = self.g.get_node(nid=pmod_id)

                                        mod_index = pmod.get("module_index")
                                        #print("interactant mod_index", mod_index, o)

                                        param_collector.append(
                                            self.get_db_index(
                                                db,
                                                mod_index,
                                                nfield_index,
                                                pindex,
                                            )
                                        )

                                        collected =True
                                        #print(f"param {pid} found in ", finid)


                                print(f"param {pid} is not in interactant -> check GHOST FIELDS")
                                for g, (gfid, gfattrs) in enumerate(ghost_fields):
                                    gikeys = gfattrs.get("keys")

                                    if isinstance(gikeys, str):
                                        print("convert ikeys", gikeys)
                                        gikeys = json.loads(gikeys)

                                    if pid in gikeys:
                                        gfield_index = gfattrs.get("field_index")
                                        print("interactant gfield_index", gfield_index, o)

                                        pindex = gikeys.index(pid)
                                        print("interactant pindex", pindex, o)

                                        # ADD PARAM TO
                                        param_collector.append(
                                            self.get_db_index(
                                                db,
                                                gmod_idx,
                                                gfield_index,
                                                pindex,
                                            )
                                        )
                                        collected = True

                                    else:
                                        print(f"param {pid} cannot be found in neighbor {gfid} for ", fid)
                                field_eq_param_struct.append(param_collector)


                            if collected is False:
                                print(f"PARAM {pid} COULD NOT BE FOUND in {fid} or its interactants... ERR")

                                # extend field with return key of method since
                                fattrs["keys"].append(pid)
                                fattrs["values"].append(None)

                                param_collector.append(
                                    self.get_db_index(
                                        db,
                                        m_idx,
                                        field_index,
                                        fattrs["keys"].index(pid),
                                    )
                                )
                                print(f"Added ghost param {pid} to {fid}")

                        # ADD EQ BLOCK TO FIELD 8 SO EACH FIELD HAS A SPACE FOR EACH EQ
                        # identify longest list
                        long = max((x for x in field_eq_param_struct if isinstance(x, list)), key=len)

                        # upscale variations list
                        res = [
                            [x if not isinstance(x, list) else x[i] for x in field_eq_param_struct]
                            for i in range(len(long))
                        ]

                        #
                        eq_param_collector.extend(res)


                        # Variations single field single eq
                        iterator["field_variations"].append(len(long))



                        # add len needed params to
                        iterator["method_param"].append(len(field_eq_param_struct))

                        self._add_return_key(eqattrs, keys, db, m_idx, field_index, hardware)

                    # add all variations for all FIELDS per EQ so, each method gets a single entry
                    # defines variation store blocks len
                    iterator["eq_variations"].append(len(eq_param_collector))



                    # unbedignt einzeln: definiert übergang nach dem variation results
                    # in feature stroe einsortiert werden
                    hardware["db_def_idx_map"].extend(eq_param_collector)

            print("compile_pattern... done")
        except Exception as e:
            print(f"Err compile_pattern: {e}")
        return hardware, iterator












    def _add_return_key(self, eqattrs, keys, db, m_idx, field_index, hardware):
        # METHOD OUT DB ENTRY
        return_key = eqattrs.get("return_key")
        print("return_key", return_key)

        ret_idx = keys.index(
            return_key
        ) if return_key in keys else 0

        # RESULT -> DB -> WORKS
        hardware["return_key_map"].append(
            self.get_db_index(db, m_idx, field_index, ret_idx)
        )
        print(f"Return Map: {return_key} -> {ret_idx}")



    def extend_fields_keys(self):
        print("extend_fields_keys...")
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        try:
            for mid, module in modules:
                if "GHOST" in mid.upper(): continue

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                if not len(methods):
                    print("compile_pattern... len methods 0")
                    continue

                print("compile_pattern meq", type(methods))

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                if len(list(fields.keys())) == 0: continue

                for eqid, eqattrs in methods.items():

                    # PARAMS from METHOD
                    return_key = eqattrs.get("return_key", [])

                    for fid, fattrs in fields.items():
                        if isinstance(fattrs["keys"], str):
                            fattrs["keys"] = json.loads(fattrs["keys"])

                        if isinstance(fattrs["values"], str):
                            fattrs["values"] = json.loads(fattrs["values"])

                        # extend field with return key of method since
                        if return_key not in fattrs["keys"]:
                            fattrs["keys"].append(return_key)
                            fattrs["values"].append([])

                        result = self.add_field_index(fid)
                        if result:
                            key, value = result
                            fattrs["keys"].append(key)
                            fattrs["values"].append(value)

        except Exception as e:
            print("Err extend_fields_keys", e)
        print("extend_fields_keys... done")




    def get_modules_methods(self, mid):
        methods = {}
        method_nodes = self.g.get_neighbor_list_rel(
            trgt_rel="has_method",
            node=mid,
            as_dict=True,
        )
        methods.update(method_nodes)

        params = self.g.get_neighbor_list_rel(
            trgt_rel="has_param",
            node=mid,
            as_dict=True,
        )

        for pid, pattrs in params.items():
            if "code" in pattrs:
                # param
                methods[pid] = pattrs



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


    def get_empty_field_structure(self, include_ghost_mod=True):
        modules: list = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )
        try:
            # 1. Calculate Max Module Index
            for mid, m in modules:
                print(f">>>module {mid}", m["module_index"])

            #nit with size max_index + 1
            modules_struct = []
            print("modules_struct initialized size:", len(modules_struct))

            for i, (mid, m) in enumerate(modules):
                if not include_ghost_mod:
                    if "GHOST" in mid.upper(): continue

                print(f"m_idx for {mid}:{i}")
                if i is None:
                    continue

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                print(f"get_empty_field_structure fields for {mid}")

                field_struct = []
                for _ in range(len(fields)):
                    field_struct.append([])

                print(f"field_struct created for {mid}")

                # SET EMPTY FIELDS STRUCT AT MODULE INDEX
                modules_struct.append(field_struct)
            pprint.pp(modules_struct)
            print("get_empty_field_structure... done")
            return modules_struct
        except Exception as e:
            print("Err get_empty_field struct:", e)


    def get_empty_method_structure(self, include_ghost_mod=True, set_none=False):
        modules: list = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )
        try:
            # 1. Calculate Max Module Index
            for midx, (mid, m) in enumerate(modules):
                print(f">>>module {mid}", midx)

            #nit with size max_index + 1
            modules_struct = [
                [] for _ in range(len(modules))
            ]
            print("modules_struct initialized size:", len(modules_struct))

            for i, (mid, m) in enumerate(modules):
                print("get_empty_method_structure", mid)
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

            print("modules_struct... done")
            return modules_struct
        except Exception as e:
            print("Err get_empty_field struct:", e)


    def get_db_index(self, db, module_idx, field_idx, param_in_field_idx):
        """
        Berechnet den globalen Index in der flachen DB-Liste.

        Args:
            db: Die verschachtelte Liste [ [field0, field1], [field0, ...] ]
            module_idx: Index des Moduls
            field_idx: Index des Feldes innerhalb des Moduls
            param_in_field_idx: Index des spezifischen Parameters im Feld-Array
        """
        # 1. Offset aller Module vor dem Ziel-Modul
        # (Summe der Längen aller Felder in diesen Modulen)
        offset_modules = sum(len(f) for m in db[:module_idx] for f in m)

        # 2. Offset aller Felder im Ziel-Modul vor dem Ziel-Feld
        offset_fields = sum(len(f) for f in db[module_idx][:field_idx])

        # 3. Der finale Index
        db_param_idx = offset_modules + offset_fields + param_in_field_idx

        return db_param_idx






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

"""

            hardware["db_def_idx_map"].extend(
                o
                for item in graph_struct["DB_OUT_GNN"]
                for o in item

            # len FEATUREs per EQ -> add int describes equation variations for field
            # eac int gets a list
            iterator["features"].append(
                max(len(sublist) for sublist in field_eq_param_struct)
            )
        
    def compile_pattern(self):
        print("compile_pattern...")

        method_out_db = []

        # structure to transfer feature to origin in gnn
        method_out_gnn = [] # just empty list because features gets index based passed

        # structure for parameter to equations -> do in gnn package (just send requested params)
        gnn_out_method = self.get_empty_field_structure()

        # db out gnn
        db_out_gnn = self.get_empty_field_structure(include_ghost_mod=False)

        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        mod_len_exclude_ghost = len(modules)-1
        print("mod_len_exclude_ghost", mod_len_exclude_ghost)

        print("get shost mods...")
        ghost_mod = self.g.get_node(nid="GHOST_MODULE")
        print("ghost_mod:", ghost_mod)

        gmod_idx = ghost_mod.get("module_index")
        print("gmod_idx:", gmod_idx)

        ghost_fields = self.g.get_neighbor_list_rel(
            trgt_rel="has_field",
            node="GHOST_MODULE",
        )
        print("ghost_fields:", len(ghost_fields))

        try:
            print("start compilation...")
            for m_idx, (mid, module) in enumerate(modules):
                print("compile_pattern... working", mid)
                if "GHOST" in mid.upper(): continue

                print("compile_pattern... module_index", m_idx)

                if m_idx is None:
                    print(f"Skipping module {mid} without index")
                    continue

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                if not len(methods):
                    print("compile_pattern... len methods 0")
                    continue

                print("compile_pattern meq", type(methods))

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                print("fields:", len(list(fields.keys())))
                if len(list(fields.keys())) == 0: continue

                for field_index, (fid, fattrs) in enumerate(fields.items()):
                    print(f"Field: {fid}", field_index)
                    # Space to save all variations for all inteactions for all equations
                    field_eq_param_struct = []
                    keys: list[str] or str = fattrs.get("keys", [])

                    if isinstance(keys, str):
                        keys = json.loads(keys)
                    print("keys:", keys)

                    fneighbors = self.g.get_neighbor_list_rel(
                        node=fid,
                        trgt_rel="has_finteractant",
                        as_dict=True
                    )


                    # Iterate Equations
                    for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                        # print("eqattrs", eqattrs)
                        print(f"Equation: {eqid}")

                        eq_param_collector = []

                        # PARAMS from METHOD
                        params = eqattrs.get("params", [])
                        if isinstance(params, str):
                            params = json.loads(params)
                        print(f"Params: {len(params)}")

                        # LOOP EQ-PARAMS
                        for pidx, pid in enumerate(params):
                            param_collector = []
                            print("work pid", pid)

                            # SELF PARAM (Field's own module)
                            if pid in keys:
                                print(f"{pid} in {fid}")
                                pindex = keys.index(pid)

                                eq_param_collector.append(
                                    [
                                        m_idx,
                                        field_index,
                                        pindex,
                                    ]
                                )
                                print(f"Mapped Self Param: {pid} -> {[m_idx, pindex, field_index]}")

                            else:
                                for o, (finid, fiattrs) in enumerate(fneighbors.items()):
                                    print("check interactant fnid", finid, o)
                                    ikeys = fiattrs.get("keys")

                                    if isinstance(ikeys, str):
                                        print("convert ikeys", ikeys)
                                        ikeys = json.loads(ikeys)

                                    # param key in interactant field?
                                    if pid in ikeys:
                                        # collect maps for all interactants
                                        print("Key in interactant")
                                        print("interactant pid", pid, o)

                                        nfield_index = fiattrs.get("field_index")
                                        print("interactant nfield_index", nfield_index, o)

                                        pindex = ikeys.index(pid)
                                        print("interactant pindex", pindex, o)

                                        # Get neighbor field's module to get its index
                                        pmod_id = fiattrs["module_id"]
                                        print("interactant pmod_id", pmod_id, o)

                                        pmod = self.g.get_node(nid=pmod_id)
                                        print("interactant pmod", pmod, o)

                                        mod_index = pmod.get("module_index")
                                        print("interactant mod_index", mod_index, o)

                                        param_collector.append(
                                            [
                                                mod_index,
                                                nfield_index,
                                                pindex,
                                            ]
                                        )


                                print(f"param {pid} is not in interactant -> check GHOST FIELDS")
                                for g, (gfid, gfattrs) in enumerate(ghost_fields):
                                    gikeys = gfattrs.get("keys")

                                    if isinstance(gikeys, str):
                                        print("convert ikeys", gikeys)
                                        gikeys = json.loads(gikeys)

                                    if pid in gikeys:
                                        gfield_index = gfattrs.get("field_index")
                                        print("interactant gfield_index", gfield_index, o)

                                        pindex = gikeys.index(pid)
                                        print("interactant pindex", pindex, o)

                                        # ADD PARAM TO
                                        eq_param_collector.append(
                                            [
                                                gmod_idx,
                                                gfield_index,
                                                pindex,
                                            ]
                                        )
                                        break
                                    else:
                                        print(f"param {pid} cannot be found")

                                # EQ VARIATION TO BLOCK COLLECTOR
                                eq_param_collector.append(param_collector)

                        # ADD EQ BLOCK TO FIELD 8 SO EACH FIELD HAS A SPACE FOR EACH EQ
                        field_eq_param_struct.append(eq_param_collector)

                        # FEATURE TO GNN -> FINISHED -> ignore equation layer when save features -> only hing matters are fdims
                        method_out_entry = [m_idx, field_index, len(field_eq_param_struct)-1]
                        print("method_out_entry:", method_out_entry)

                        # APPLY FEATURE WORKS
                        method_out_gnn.append(method_out_entry)
                        print("method_out_gnn:", method_out_gnn)

                        # METHOD OUT DB ENTRY
                        return_key = eqattrs.get("return_key")
                        print("return_key", return_key)

                        ret_idx = keys.index(
                            return_key
                        ) if return_key in keys else 0

                        # RESULT -> DB -> WORKS
                        method_out_db.append([m_idx, field_index, ret_idx])
                        print(f"Return Map: {return_key} -> {ret_idx}")

                    # PARAM PATHWAY METHOD VARIATION # -> WORKS
                    # mach es doch auf eq ebene -> so hat jeder eq layer die selbe länge
                    db_out_gnn[m_idx][field_index] = field_eq_param_struct
                    print("db_out_gnn entry saved")

            print("compile_pattern... done")
        except Exception as e:
            print(f"Err compile_pattern: {e}")

















    def model_skeleton(self, env_id, db):

model = {
    "DB": [], # db param indices
    "DB_TO_FEATURES": [], # link just input to feature
    "DEF_FEATURES": [], # node to collect features
    "FEATURES_TO_DB":[], # link just output of each calculation to db
}


modules = self.g.get_neighbor_list(
    node=env_id,
    target_type="MODULE",
)

# horizontale untertielung: 5f * [[3] - [2]]
for midx, mattrs in enumerate(modules):

    fields = self.g.get_neighbor_list_rel(
        node=mattrs["nid"],
        trgt_rel="has_field",
        as_dict=True,
    )

    for fidx in range(len(list(fields.keys()))):




    def compile_pattern(self):
        print("compile_pattern...")

        method_out_db = self.get_empty_field_structure()

        # structure to transfer feature to origin in gnn
        method_out_gnn = [] # just empty list because features gets index based passed

        # structure for parameter to equations -> do in gnn package (just send requested params)
        gnn_out_method = self.get_empty_field_structure()

        # db out gnn
        db_out_gnn = self.get_empty_field_structure()

        # MAP TO FEATURES
        feature_store_struct = self.get_empty_field_structure()

        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        mod_len_exclude_ghost = len(modules)-1
        print("mod_len_exclude_ghost", mod_len_exclude_ghost)


        print("get shost mods...")
        ghost_mod = self.g.get_node(nid="GHOST_MODULE")
        print("ghost_mod:", ghost_mod)

        gmod_idx = ghost_mod.get("module_index")
        print("gmod_idx:", gmod_idx)

        ghost_fields = self.g.get_neighbor_list_rel(
            trgt_rel="has_field",
            node="GHOST_MODULE",
        )
        print("ghost_fields:", len(ghost_fields))

        try:
            print("start compilation...")
            for m_idx, (mid, module) in enumerate(modules):
                print("compile_pattern... working", mid)
                if "GHOST" in mid.upper(): continue

                print("compile_pattern... module_index", m_idx)

                if m_idx is None:
                    print(f"Skipping module {mid} without index")
                    continue

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                if not len(methods):
                    print("compile_pattern... len methods 0")
                    continue

                print("compile_pattern meq", type(methods))

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                print("fields:", len(list(fields.keys())))

                if len(list(fields.keys())) == 0: continue

                # Iterate Equations
                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    # print("eqattrs", eqattrs)
                    print(f"Equation: {eqid}")

                    # PARAMS from METHOD
                    params = eqattrs.get("params", [])
                    if isinstance(params, str):
                        params = json.loads(params)
                    print(f"Params: {len(params)}")

                    for field_index, (fid, fattrs) in enumerate(fields.items()):
                        print(f"Field: {fid}")
                        keys: list[str] or str = fattrs.get("keys", [])

                        print("field index:", field_index)

                        if isinstance(keys, str):
                            keys = json.loads(keys)
                        print("keys:", keys)

                        fneighbors = self.g.get_neighbor_list_rel(
                            node=fid,
                            trgt_rel="has_finteractant",
                            as_dict=True
                        )

                        if field_index is None: continue

                        for o, (finid, fiattrs) in enumerate(fneighbors.items()):
                            field_param_struct = []

                            print("check interactant fnid", finid, o)
                            ikeys = fiattrs.get("keys")

                            if isinstance(ikeys, str):
                                print("convert ikeys", ikeys)
                                ikeys = json.loads(ikeys)

                            # LOOP EQ-PARAMS
                            for pidx, pid in enumerate(params):
                                print("work pid", pid)

                                # SELF PARAM (Field's own module)
                                if pid in keys:
                                    print(f"{pid} in {fid}")
                                    pindex = keys.index(pid)

                                    field_param_struct.append(
                                        [
                                            m_idx,
                                            field_index,
                                            pindex,
                                        ]
                                    )
                                    print(f"Mapped Self Param: {pid} -> {[m_idx, pindex, field_index]}")

                                # param key in interactant field?
                                elif pid in ikeys:
                                    # collect maps for all interactants
                                    print("Key in interactant")
                                    print("interactant pid", pid, o)

                                    nfield_index = fiattrs.get("field_index")
                                    print("interactant nfield_index", nfield_index, o)

                                    pindex = ikeys.index(pid)
                                    print("interactant pindex", pindex, o)

                                    # Get neighbor field's module to get its index
                                    pmod_id = fiattrs["module_id"]
                                    print("interactant pmod_id", pmod_id, o)

                                    pmod = self.g.get_node(nid=pmod_id)
                                    print("interactant pmod", pmod, o)

                                    mod_index = pmod.get("module_index")
                                    print("interactant mod_index", mod_index, o)

                                    field_param_struct.append(
                                        [
                                            mod_index,
                                            nfield_index,
                                            pindex,
                                        ]
                                    )
                                else:
                                    print(f"param {pid} is not in interactant -> check GHOST FIELDS")
                                    for g, (gfid, gfattrs) in enumerate(ghost_fields):
                                        gikeys = gfattrs.get("keys")

                                        if isinstance(gikeys, str):
                                            print("convert ikeys", gikeys)
                                            gikeys = json.loads(gikeys)

                                        if pid in gikeys:
                                            gfield_index = gfattrs.get("field_index")
                                            print("interactant gfield_index", gfield_index, o)

                                            pindex = gikeys.index(pid)
                                            print("interactant pindex", pindex, o)

                                            # ADD PARAM TO
                                            field_param_struct.append(
                                                [
                                                    gmod_idx,
                                                    gfield_index,
                                                    pindex,
                                                ]
                                            )
                                        else:
                                            print(f"param {pid} cannot be found")

                            # FEATURE TO GNN
                            method_out_entry = [m_idx, eq_idx, field_index]
                            print("method_out_entry:", method_out_entry)

                            method_out_gnn.append(method_out_entry)
                            print("method_out_gnn:", method_out_gnn)

                            # METHOD OUT DB ENTRY
                            return_key = eqattrs.get("return_key")
                            print("return_key", return_key)

                            ret_idx = keys.index(
                                return_key
                            ) if return_key in keys else 0

                            #method_out_db[m_idx][field_index].append(ret_idx)
                            method_out_db.append([m_idx, field_index, ret_idx])
                            print(f"Return Map: {return_key} -> {ret_idx}")

                            # CREATE FEATURE DIM FOR EACH METHOD INPOUT VARIATION / FIELD
                            feature_store_struct.append([m_idx, eq_idx, field_index])
                            print("feature_store_struct entry saved")

                            # PARAM PATHWAY METHOD VARIATION
                            print("db_out_gnn entry m_idx,eq_idx,field_index", m_idx,eq_idx,field_index, db_out_gnn)

                            db_out_gnn[
                                m_idx][field_index
                            ].append(field_param_struct)
                            print("db_out_gnn entry saved")

            print("compile_pattern... done")
        except Exception as e:
            print(f"Err compile_pattern: {e}")

        return {
            "METHOD_OUT_DB": method_out_db,
            "FEATURE_OUT_GNN": method_out_gnn,
            "GNN_OUT_METHOD": gnn_out_method,
            "DB_OUT_GNN": db_out_gnn,
            #"FEATURE_OUT_GNN": feature_store_struct
        }


        try:
            print("start compilation...")
            for m_idx, (mid, module) in enumerate(modules):
                print("compile_pattern... working", mid)
                if "GHOST" in mid.upper(): continue

                print("compile_pattern... module_index", m_idx)

                if m_idx is None:
                    print(f"Skipping module {mid} without index")
                    continue

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                if not len(methods):
                    print("compile_pattern... len methods 0")
                    continue

                print("compile_pattern meq", type(methods))

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                print("fields:", len(list(fields.keys())))
                if len(list(fields.keys())) == 0: continue

                for field_index, (fid, fattrs) in enumerate(fields.items()):
                    print(f"Field: {fid}", field_index)

                    # Space to save all variations for all inteactions for all equations
                    field_eq_param_struct = []
                    keys: list[str] or str = fattrs.get("keys", [])

                    if isinstance(keys, str):
                        keys = json.loads(keys)
                    print("keys:", keys)

                    fneighbors = self.g.get_neighbor_list_rel(
                        node=fid,
                        trgt_rel="has_finteractant",
                        as_dict=True
                    )

                    # einzelne nicht verschachtelte loops :


                    # Iterate Equations
                    for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):

                        # print("eqattrs", eqattrs)
                        print(f"Equation: {eqid}")

                        eq_param_collector = []

                        # PARAMS from METHOD
                        params = eqattrs.get("params", [])
                        if isinstance(params, str):
                            params = json.loads(params)
                        print(f"Params: {len(params)}")

                        # LOOP EQ-PARAMS
                        for pidx, pid in enumerate(params):
                            #field_param_struct = []
                            print("work pid", pid)

                            for o, (finid, fiattrs) in enumerate(fneighbors.items()):
                                print("check interactant fnid", finid, o)
                                ikeys = fiattrs.get("keys")

                                if isinstance(ikeys, str):
                                    print("convert ikeys", ikeys)
                                    ikeys = json.loads(ikeys)

                                # SELF PARAM (Field's own module)
                                if pid in keys:
                                    print(f"{pid} in {fid}")
                                    pindex = keys.index(pid)

                                    eq_param_collector.append(
                                        [
                                            m_idx,
                                            field_index,
                                            pindex,
                                        ]
                                    )
                                    print(f"Mapped Self Param: {pid} -> {[m_idx, pindex, field_index]}")
                                    continue

                                # param key in interactant field?
                                elif pid in ikeys:
                                    # collect maps for all interactants
                                    print("Key in interactant")
                                    print("interactant pid", pid, o)

                                    nfield_index = fiattrs.get("field_index")
                                    print("interactant nfield_index", nfield_index, o)

                                    pindex = ikeys.index(pid)
                                    print("interactant pindex", pindex, o)

                                    # Get neighbor field's module to get its index
                                    pmod_id = fiattrs["module_id"]
                                    print("interactant pmod_id", pmod_id, o)

                                    pmod = self.g.get_node(nid=pmod_id)
                                    print("interactant pmod", pmod, o)

                                    mod_index = pmod.get("module_index")
                                    print("interactant mod_index", mod_index, o)

                                    eq_param_collector.append(
                                        [
                                            mod_index,
                                            nfield_index,
                                            pindex,
                                        ]
                                    )
                                    break
                                else:
                                    print(f"param {pid} is not in interactant -> check GHOST FIELDS")
                                    for g, (gfid, gfattrs) in enumerate(ghost_fields):
                                        gikeys = gfattrs.get("keys")

                                        if isinstance(gikeys, str):
                                            print("convert ikeys", gikeys)
                                            gikeys = json.loads(gikeys)

                                        if pid in gikeys:
                                            gfield_index = gfattrs.get("field_index")
                                            print("interactant gfield_index", gfield_index, o)

                                            pindex = gikeys.index(pid)
                                            print("interactant pindex", pindex, o)

                                            # ADD PARAM TO
                                            eq_param_collector.append(
                                                [
                                                    gmod_idx,
                                                    gfield_index,
                                                    pindex,
                                                ]
                                            )
                                            break
                                        else:
                                            print(f"param {pid} cannot be found")

                            # COLLECT METHOD PARAM-STRUCT IN EQ SPACE
                            eq_param_collector.append(field_param_struct)

                        # ADD EQ_SPACE TO FIELD 8 SO EACH FIELD HAS A SPACE FOR EACH EQ
                        field_eq_param_struct.append(eq_param_collector)

                        # FEATURE TO GNN -> FINISHED -> ignore equation layer when save features -> only hing matters are fdims
                        method_out_entry = [m_idx, field_index, len(field_eq_param_struct)-1]
                        print("method_out_entry:", method_out_entry)

                        # APPLY FEATURE WORKS
                        method_out_gnn.append(method_out_entry)
                        print("method_out_gnn:", method_out_gnn)

                        # METHOD OUT DB ENTRY
                        return_key = eqattrs.get("return_key")
                        print("return_key", return_key)

                        ret_idx = keys.index(
                            return_key
                        ) if return_key in keys else 0

                        # RESULT -> DB -> WORKS
                        method_out_db.append([m_idx, field_index, ret_idx])
                        print(f"Return Map: {return_key} -> {ret_idx}")

                    # PARAM PATHWAY METHOD VARIATION # -> WORKS
                    # mach es doch auf eq ebene -> so hat jeder eq layer die selbe länge
                    db_out_gnn[m_idx][field_index] = field_eq_param_struct
                    print("db_out_gnn entry saved")

            print("compile_pattern... done")
        except Exception as e:
            print(f"Err compile_pattern: {e}")


    def compile_pattern(self):
        print("compile_pattern...")
        # todo for field index includ again

        graph_struct = {
            "METHOD_OUT_DB": [],
            "FEATURE_SKELETON": [],
            "DB_OUT_GNN": [],
        }

        # db out gnn
        #db_out_gnn = self.get_empty_field_structure(include_ghost_mod=False)

        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        mod_len_exclude_ghost = len(modules)-1
        print("mod_len_exclude_ghost", mod_len_exclude_ghost)

        print("get shost mods...")
        ghost_mod = self.g.get_node(nid="GHOST_MODULE")
        print("ghost_mod:", ghost_mod)

        gmod_idx = ghost_mod.get("module_index")
        print("gmod_idx:", gmod_idx)

        ghost_fields = self.g.get_neighbor_list_rel(
            trgt_rel="has_field",
            node="GHOST_MODULE",
        )
        print("ghost_fields:", len(ghost_fields))


        iterator = {
            # GOAL FOR THE ITERATOR COMPONENT
            # map int to feature
            # map feature -> field (&field -> mod)
            # e.g. given index 1000 : len features * sum(features = amount all features // (sum(fields) / len(fields)) = field index

            # index space for equation len per module
            "modules": [], # save len eq for each module

            # index space len fields per module (why module? because each module and underlying equation uses same fields (in this model)
            "fields": [],

            # space amount features
            "features": [],

            # space amount params per feature
            "db": [],
        }

        # g1 5f
        try:
            print("start compilation...")
            for m_idx, (mid, module) in enumerate(modules):
                print("compile_pattern... working", mid)
                if "GHOST" in mid.upper(): continue

                print("compile_pattern... module_index", m_idx)

                if m_idx is None:
                    print(f"Skipping module {mid} without index")
                    continue

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                if not len(list(methods.keys())):
                    print("compile_pattern... len methods 0")
                    continue

                # Lets append the len of the methods to the code struct
                iterator["modules"].append(len(list(methods.keys())))

                print("compile_pattern meq", type(methods))

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )


                len_fields = len(list(fields.keys()))
                print("fields:", len_fields)
                if len(list(fields.keys())) == 0: raise Exception("Err: no fields found...")
                iterator["fields"].append(len_fields)

                # MARK: wenn wir gleichungen als äußerstes element haben dann erstellen wir automatische einen bidirketionalen
                # Graphen (eq->field->param_struct zeigt auf struct mit param mappings zu midx->field->param)

                # featurs dürfen nicht gemerget werden -> da direkte signal Verfolgung sons verloren geht (unter geht)
                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    # print("eqattrs", eqattrs)
                    print(f"Equation: {eqid}")

                    # field param blocks collector
                    eq_param_collector = []
                    return_param_map = []

                    # PARAMS from METHOD
                    params = eqattrs.get("params", [])
                    print("methods params", params)

                    # PARAMS from METHOD
                    return_key = eqattrs.get("return_key", [])
                    print(f"methods {eqid} return_key", return_key)

                    # params_origin
                    params_origin = eqattrs.get("origin", None)

                    if params_origin is None:
                        params_origin = ["" for _ in range(len(params))]
                    print("methods params_origin", params_origin)

                    if isinstance(params, str):
                        params = json.loads(params)
                    print(f"Params: {len(params)}")

                    for field_index, (fid, fattrs) in enumerate(fields.items()):
                        #print(f"Field: {fid}", field_index)

                        fidx = module.get("field_index")
                        print("fidx", fidx)

                        # Space to save all variations for all inteactions for all equations
                        field_eq_param_struct = []

                        if isinstance(fattrs["keys"], str):
                            fattrs["keys"] = json.loads(fattrs["keys"])

                        if isinstance(fattrs["values"], str):
                            fattrs["values"] = json.loads(fattrs["values"])

                        keys: list[str] or str = fattrs.get("keys", [])

                        if isinstance(keys, str):
                            keys = json.loads(keys)
                        print(f"{fid} keys:", keys)

                        fneighbors = self.g.get_neighbor_list_rel(
                            node=fid,
                            trgt_rel="has_finteractant",
                            as_dict=True
                        )

                        # LOOP EQ-PARAMS
                        for pidx, pid in enumerate(params):
                            collected = False

                            param_collector = []
                            print("work pid", pid)

                            # SELF PARAM (Field's own module)
                            is_prefixed = pid.endswith("_")

                            EXCLUDED_ORIGINS = ["neighbor", "interactant"]

                            if (
                                pid in keys and not is_prefixed and params_origin[pidx] not in EXCLUDED_ORIGINS
                            ):
                                print(f"{pid} in {fid}")

                                pindex = keys.index(pid)

                                field_eq_param_struct.append([
                                        m_idx,
                                        field_index,
                                        pindex,
                                    ]
                                )

                                collected=True
                                print(f"Mapped Self Param: {pid} -> {fid}")

                            else:
                                print(f"param {pid} not in {keys}")

                                for _ in range(len(pid)):
                                    if is_prefixed:
                                        print("remove slicing end char from", pid)
                                        pid = pid[:-1]
                                        print("edited pid", pid)
                                        break
                                    else:
                                        break

                                for o, (finid, fiattrs) in enumerate(fneighbors.items()):
                                    print("check interactant fnid", finid, o)
                                    ikeys = fiattrs.get("keys")

                                    if isinstance(ikeys, str):
                                        print("convert ikeys", ikeys)
                                        ikeys = json.loads(ikeys)

                                    # param key in interactant field?
                                    if pid in ikeys:
                                        # collect maps for all interactants
                                        #print("interactant pid", pid, o)

                                        nfield_index = fiattrs.get("field_index")
                                        #print("interactant nfield_index", nfield_index, o)

                                        pindex = ikeys.index(pid)
                                        #print("interactant pindex", pindex, o)

                                        # Get neighbor field's module to get its index
                                        pmod_id = fiattrs["module_id"]
                                        #print("interactant pmod_id", pmod_id, o)

                                        pmod = self.g.get_node(nid=pmod_id)

                                        mod_index = pmod.get("module_index")
                                        #print("interactant mod_index", mod_index, o)

                                        param_collector.append(
                                            [
                                                mod_index,
                                                nfield_index,
                                                pindex,
                                            ]
                                        )
                                        collected =True
                                        #print(f"param {pid} found in ", finid)


                                print(f"param {pid} is not in interactant -> check GHOST FIELDS")
                                for g, (gfid, gfattrs) in enumerate(ghost_fields):
                                    gikeys = gfattrs.get("keys")

                                    if isinstance(gikeys, str):
                                        print("convert ikeys", gikeys)
                                        gikeys = json.loads(gikeys)

                                    if pid in gikeys:
                                        gfield_index = gfattrs.get("field_index")
                                        print("interactant gfield_index", gfield_index, o)

                                        pindex = gikeys.index(pid)
                                        print("interactant pindex", pindex, o)

                                        # ADD PARAM TO
                                        param_collector.append(
                                            [
                                                gmod_idx,
                                                gfield_index,
                                                pindex,
                                            ]
                                        )

                                        collected = True

                                    else:
                                        print(f"param {pid} cannot be found in neighbor {gfid} for ", fid)
                                field_eq_param_struct.append(param_collector)

                            if collected is False:
                                print(f"PARAM {pid} COULD NOT BE FOUND in {fid} or its interactants... ERR")

                                # extend field with return key of method since
                                fattrs["keys"].append(pid)
                                fattrs["values"].append(None)

                                param_collector.append([
                                    m_idx,
                                    field_index,
                                    fattrs["keys"].index(pid),
                                ])
                                print(f"Added ghost param {pid} to {fid}")

                        # ADD EQ BLOCK TO FIELD 8 SO EACH FIELD HAS A SPACE FOR EACH EQ
                        eq_param_collector.append(field_eq_param_struct)

                        # FEATURE TO GNN -> add int describes equation variations for field
                        # eac int gets a list
                        #
                        iterator["features"].append(max(len(sublist) for sublist in field_eq_param_struct))

                        # METHOD OUT DB ENTRY
                        return_key = eqattrs.get("return_key")
                        print("return_key", return_key)

                        ret_idx = keys.index(
                            return_key
                        ) if return_key in keys else 0

                        # RESULT -> DB -> WORKS
                        return_param_map.append([m_idx, field_index, ret_idx])
                        print(f"Return Map: {return_key} -> {ret_idx}")




                    # EQ LAYER APPEND
                    graph_struct["DB_OUT_GNN"].append(eq_param_collector)

                    #
                    graph_struct["METHOD_OUT_DB"].append(return_param_map)
                    print("graph struct entry saved")

            print("compile_pattern... done")
        except Exception as e:
            print(f"Err compile_pattern: {e}")
        return graph_struct
"""






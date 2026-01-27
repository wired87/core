import base64
import json
import pprint

from itertools import product

import numpy as np
from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin
from core.env_manager import EnvManager

from core.fields_manager.fields_lib import FieldsManager
from core.method_manager.method_lib import MethodManager

from core.injection_manager import InjectionManager
from core.module_manager.ws_modules_manager import ModuleWsManager
from core.param_manager.params_lib import ParamsManager
from core.qbrain_manager import QBrainTableManager
from core.user_manager import UserManager

from core.module_manager.mcreator import ModuleCreator
from qf_utils.all_subs import ALL_SUBS
from utils._np.expand_array import expand_structure
from utils.get_shape import get_modular_shape, extract_complex
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

    todo: curretnly all dims implemented within single db inject-> create db / d whcih captures jsut sinfle point

    #
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
        print("DEBUG: ArtifactAdmin initialized")
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
                    # add clean inj_id (inj_id = id in db from pattern)
                    inj_ids.add(inj_id)

                    inj_id = f"{pos}__{inj_id}"
                    print("ADD INJECTION", inj_id)
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
            module_index=len(modules), # last module
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

        print("check_create_ghost_mod... done")


    def handle_env(self, env_id):
        print("handle_env...")

        #
        keys = list(self.qfu.create_env().keys())

        # FETCH ENV CONSTANTS
        env_constants = self.qb.row_from_id(keys, table="params")

        # VALUE
        values = [v["value"] for v in env_constants]

        # AXIS
        axis_def = [None for _ in env_constants]

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

        print("GHOST_MOD FIELDS", fields, len(fields))

        print("ADD ENV FIELD")
        self.g.add_node(
            dict(
                nid=env_id,
                type="FIELD",
                sub_type="ENV",
                keys=keys,
                values=values,
                axis_def=axis_def,
                field_index=len(fields),
            )
        )
        self.g.add_edge(
            "GHOST_MODULE",
            env_id,
            attrs={
                "rel": "has_field",
                "src_layer": "MODULE",
                "trgt_layer": "FIELD",
            }
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
                    table=self.env_manager.TABLE_ID
                )
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


    def main(self, env_id, env_data):
        self.data_handler(env_id, env_data)
        components = self.converter(env_id)
        self.handle_deployment(env_id, components)
        print("main... done")


    def handle_deployment(self, env_id, components):
        print("handle_deployment...")
        try:
            world_cfg = self.create_vm_cfgs(env_id)
            # SAVE CFG
            self.module_db_manager.qb.update_env_pattern(
                env_id=env_id,
                pattern_data=components,
                user_id=self.user_id
            )
            if self.testing is False:
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

                self.deploy_vms(vm_payload)

                # Update status
                self.user_manager.qb.set_item(
                    "envs",
                    {"status": "IN_PROGRESS"},
                    keys={"id": env_id, "user_id": self.user_id}
                )
            else:
                print("save file local...")
                with open(r"C:\Users\bestb\PycharmProjects\BestBrain\test_out.json", "w") as f:
                    f.write(json.dumps(components))
        except Exception as e:
            print("Err handle_deployment:", e)
        print("handle_deployment... done")

    def data_handler(self, env_id, env_data):
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


                    # create complex map for tbhe db indexing

                    self.g.update_node(
                        dict(
                            nid=fid,
                            type="FIELD",
                            **fattrs
                        )
                    )
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
            print(f"Processing module: {mod_id}")

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

        print("data_hanadler... done")


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


    def converter(self, env_id:str):
        """
        CREATE/COLL ECT PATTERNS FOR ALL ENVS AND CREATE VM
        """
        print("Main started...")
        env_node = self.g.get_node(env_id)
        modules: list = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        self.DB = self.create_db(
            modules,
            env_node["amount_of_nodes"]
        ) # single param lvl -> todo grid: scale vertical

        db_to_method_struct:dict = self.set_edge_db_to_method(modules)

        method_struct:dict = self.method_layer(modules)

        method_to_db = self.set_edge_method_to_db(
            modules,
        )

        injection_patterns = self.set_inj_pattern(env_node)

        # create iterator last
        iterators = self.set_iterator_from_humans()



        components = {
            **self.DB,
            **iterators,
            **method_struct,
            **injection_patterns,
            **db_to_method_struct,
            "METHOD_TO_DB": method_to_db,
        }

        print("components created")
        #pprint.pp(components)
        print("Main... done")
        return components


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
            **cfg
        ):
        env_node = self.g.get_node(env_id)
        env_node = {k:v for k,v in env_node.items() if k not in ["updated_at", "created_at"]}
        #print("add env node to world cfg:", env_node)

        # BOB BUILDER ACTION
        world_cfg = {
            **cfg,
            "ENV_ID": env_id,
            "START_TIME": env_node.get("sim_time", 1),
            "AMOUNT_NODES": env_node.get("amount_of_nodes", 1),
            "DIMS": env_node.get("dims", 3),
        }

        cfg_str = json.dumps(world_cfg)

        return cfg_str




    def get_state(self):
        return len(self.fields) and self.world_cfg



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

        INJECTOR = {
            "INJECTOR_TIME":[],
            "INJECTOR_INDICES":[],
            "INJECTOR_VALUES":[],
        }

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
                            field_rel_param_trgt_index = f_keys.index(key_opt)

                            # 5. Loop Injections
                            injections = self.g.get_neighbor_list_rel(
                                node=fid,
                                trgt_rel="has_injection",
                            )

                            if len(injections) > 0:
                                print(f"INJs for FIELD {fid}:", len(injections))

                            if not injections or not len(injections):
                                continue

                            for inj_id, inj_attrs in injections:
                                print("work inj_id:", inj_id)

                                # set the index within the extracted value slice of the 1d db

                                pos_index_slice:int or None = schema_positions.index(
                                    tuple(eval(inj_id.split("__")[0]))
                                )


                                ### BAUSTELLE
                                for time, data in zip(inj_attrs["data"][0], inj_attrs["data"][1]):
                                    if time not in INJECTOR["INJECTOR_TIME"]:
                                        INJECTOR["INJECTOR_TIME"].append(time)
                                        INJECTOR["INJECTOR_INDICES"].append([])
                                        INJECTOR["INJECTOR_VALUES"].append([])
                                        
                                    tidx = INJECTOR["INJECTOR_TIME"].index(time)
                                    INJECTOR["INJECTOR_INDICES"][tidx].append((midx, fi, field_rel_param_trgt_index, pos_index_slice))
                                    INJECTOR["INJECTOR_VALUES"][tidx].append(data)


                                print(
                                    f"set param pathway db from mod {midx} -> field {fattrs['nid']}({fi})"
                                )
                            break
            print(f"set_inj_pattern... done")
        except Exception as e:
            print("Err set_inj_pattern", e)
        return INJECTOR




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




    def create_db(self, modules, amount_nodes):
        """
        Collect all node values and stack them into a flat DB
        -> uniform layout for all environments
        """
        print("[create_db] start")

        # --- initialize base structures ---
        db = self.get_empty_field_structure()
        axis = self.get_empty_field_structure()
        shapes = self.get_empty_field_structure()
        item_len_collection = self.get_empty_field_structure()
        param_len_collection = self.get_empty_field_structure()

        # flat DB dict
        DB = {
            "DB": [],
            "AXIS": [],
            "DB_SHAPE": [],
            "AMOUNT_PARAMS_PER_FIELD": [],
            "DB_PARAM_CONTROLLER": [],
        }

        try:
            # --- iterate modules ---
            for mid, m in modules:
                m_idx = m.get("module_index")

                print(f"[create_db] module={mid}, idx={m_idx}")

                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True
                )

                keys = list(fields.keys())
                print(f">>>{mid} create_db FIELDS", keys, len(keys))
                # --- iterate fields ---
                for fid, fattrs in fields.items():
                    fidx = fattrs.get("field_index")
                    print(f"[field] apply pattern -> module={m_idx}, field={fidx}")

                    # axis definition
                    xdef = fattrs.get("axis_def", [])
                    if isinstance(xdef, str):
                        xdef = json.loads(xdef)

                    # values
                    vals = fattrs.get("values", [])
                    if isinstance(vals, str):
                        vals = json.loads(vals)

                    shape_struct = []
                    collected_values = []
                    vals_item_lens = []

                    # --- process values ---
                    for vi, v in enumerate(vals):
                        shape = get_modular_shape(v)
                        shape_struct.append(shape)

                        # flatten any shape
                        val_len = []
                        extract_complex(v, collected_values, val_len)
                        vals_item_lens.append(
                            len(val_len)
                        )

                        print(f"[val {vi}] shape={shape}, len={len(val_len)}")

                    # --- assign per module / field ---
                    db[m_idx][fidx] = collected_values
                    axis[m_idx][fidx] = xdef
                    shapes[m_idx][fidx] = shape_struct
                    item_len_collection[m_idx][fidx] = vals_item_lens
                    param_len_collection[m_idx][fidx] = len(vals)


            for m_db, m_axis, m_shape, m_len, plen_item in zip(db, axis, shapes, item_len_collection, param_len_collection):
                for f_db, f_axis, f_shape, f_len, plen_field in zip(m_db, m_axis, m_shape, m_len, plen_item):
                    DB["DB"].extend(f_db)
                    DB["AXIS"].extend(f_axis)
                    DB["DB_SHAPE"].extend(f_shape)

                    # int len single param in db unscaled (e.g. 3 for "000")
                    DB["DB_PARAM_CONTROLLER"].extend(f_len)
                    # AMOUNT_PARAMS_PER_FIELD
                    # int len param / f
                    DB["AMOUNT_PARAMS_PER_FIELD"].append(plen_field)

            # --- finalize DB ---
            print("DB before serialization:", DB["DB"])

            DB["DB"] = base64.b64encode(
                np.array(DB["DB"], dtype=np.complex64).tobytes()
            ).decode("utf-8")

            print("[create_db] done")

        except Exception as e:
            print("[create_db][ERROR]", e)
        return DB



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



    def method_layer(self, modules):
        print("method_layer... ")

        mod_len_exclude_ghost = len(modules) -1
        print("method_layer mod_len_exclude_ghost", mod_len_exclude_ghost)
        method_struct = {
            "METHOD_PARAM_LEN_CTLR": [[] for _ in range(mod_len_exclude_ghost)],
            "METHODS": [[] for _ in range(mod_len_exclude_ghost)],
            "METHODS_PER_MOD_LEN_CTLR": [0 for _ in range(mod_len_exclude_ghost)],
        }

        try:
            print("method_layer compilation...")
            for mid, module in modules:
                # ghost does not have equation
                if "GHOST" in mid.upper(): continue
                print("method_layer... working", mid)

                midx = module.get("module_index")
                print("midx", midx)

                print("method_layer... module_index")

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                mlen = len(list(methods.keys()))
                if not mlen:
                    print("method_layer... len methods", mlen)
                    continue

                print("method_layer eqs", type(methods))

                # len methods per module
                method_struct["METHODS_PER_MOD_LEN_CTLR"][midx] = mlen

                # Iterate Equations
                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    # PARAMS from METHOD
                    params = eqattrs.get("params", [])
                    if isinstance(params, str):
                        params = json.loads(params)
                    print(f"Params: {len(params)}")

                    method_struct["METHOD_PARAM_LEN_CTLR"][midx].append(
                        len(params)
                    )

                    method_struct["METHODS"][midx].append(
                        eqattrs.get("jax_code", eqattrs.get("code"))
                    )

            # flatten
            flattened_methods = []
            for sublist in method_struct["METHODS"]:
                flattened_methods.extend(sublist)
            method_struct["METHODS"] = flattened_methods

            flatten_ctlr = []
            for sublist in method_struct["METHOD_PARAM_LEN_CTLR"]:
                flatten_ctlr.extend(sublist)
            method_struct["METHOD_PARAM_LEN_CTLR"] = flatten_ctlr

        except Exception as e:
            print("Err method_layer", e)
        pprint.pp(method_struct)
        print(f"method_layer... done")
        return method_struct


    def set_edge_method_to_db(self, modules):
        # each eqs fields has different return key (IMPORTANT: sum variation results)
        return_key_map = []
        try:
            print("start compilation...")
            for m_idx, (mid, module) in enumerate(modules):
                print("set_edge_method_to_db... working", mid)
                if "GHOST" in mid.upper(): continue

                print("set_edge_db_to_method... module_index", m_idx)

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
                    print("set_edge_db_to_method... len methods 0")
                    continue

                #print("set_edge_db_to_method meq", type(methods))

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    # print("eqattrs", eqattrs)
                    print(f"Equation: {eqid}")

                    for field_index, (fid, fattrs) in enumerate(fields.items()):
                        keys: list[str] or str = fattrs.get("keys", [])

                        return_key = eqattrs.get("return_key")

                        rindex = keys.index(return_key)

                        fmod = self.g.get_node(nid=fattrs["module_id"])

                        # RESULT -> DB -> WORKS
                        return_key_map.append(
                            self.get_db_index(
                                fmod["module_index"],
                                field_index,
                                rindex,
                            )
                        )
                        #print(f"Return Map: {return_key} -> {ret_idx}")
        except Exception as e:
            print("Err set_edge_method_to_db", e)
            pass
        return return_key_map


    def set_edge_db_to_method(self, modules):
        print("set_edge_db_to_method...")
        # todo for field index includ again
        mlen = len(modules)-1
        # db out gnn
        db_to_method = {
            "DB_TO_METHOD_EDGES": self.get_empty_method_structure(set_zero=False),
            "DB_CTL_VARIATION_LEN_PER_EQUATION": [[] for _ in range(mlen)],
        }

        # um Variationen und Gleichung zusammen zu mappen brauchen wir
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
                if "GHOST" in mid.upper(): continue
                print("set_edge_db_to_method... working", mid)


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
                    print("set_edge_db_to_method... len methods 0")
                    continue

                print("set_edge_db_to_method meq", type(methods))
                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                len_fields = len(list(fields.keys()))
                print("fields:", len_fields)
                if len(list(fields.keys())) == 0: raise Exception("Err: no fields found...")

                # featurs dürfen nicht gemerget werden -> da direkte signal Verfolgung sons verloren geht (unter geht)
                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    # print("eqattrs", eqattrs)
                    print(f"Equation: {eqid}")
                    EQ_AMOUNT_VRIATIONS = 0
                    # PARAMS from METHOD
                    params = eqattrs.get("params", [])
                    print("methods params", params)

                    # params_origin
                    params_origin = eqattrs.get("origin", None)

                    if params_origin is None:
                        params_origin = ["" for _ in range(len(params))]
                    print("methods params_origin", params_origin)

                    if isinstance(params, str):
                        params = json.loads(params)
                    #print(f"Params: {len(params)}")

                    for field_index, (fid, fattrs) in enumerate(fields.items()):

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
                                        module["module_index"],
                                        field_index,
                                        pindex,
                                    )
                                )

                                collected=True

                                #print(f"Mapped Self Param: {pid} -> {fid}")

                            else:
                                print(f"param {pid} not in {fid} keys")

                                pid = pid[:-1] if is_prefixed else pid

                                for o, (finid, fiattrs) in enumerate(fneighbors.items()):
                                    #print("check interactant fnid", finid, o)
                                    ikeys = fiattrs.get("keys")

                                    if isinstance(ikeys, str):
                                        #print("convert ikeys", ikeys)
                                        ikeys = json.loads(ikeys)

                                    # param key in interactant field?
                                    if pid in ikeys:
                                        fmod = self.g.get_node(nid=fiattrs.get("module_id"))

                                        # collect maps for all interactants
                                        #print("interactant pid", pid, o)

                                        nfield_index = fiattrs.get("field_index")

                                        pindex = ikeys.index(pid)
                                        #print("interactant pindex", pindex, o)

                                        param_collector.append(
                                            self.get_db_index(
                                                fmod["module_index"],
                                                nfield_index,
                                                pindex,
                                            )
                                        )
                                        collected =True


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

                                        gmod = self.g.get_node(nid="GHOST_MODULE")
                                        # ADD PARAM TO
                                        param_collector.append(
                                            self.get_db_index(
                                                gmod["module_index"],
                                                gfield_index,
                                                pindex,

                                            )
                                        )
                                        collected = True

                                    else:
                                        print(f"param {pid} cannot be found in neighbor {gfid} for ", fid)

                                    #
                                    if collected is True:
                                        field_eq_param_struct.append(param_collector)

                            if collected is False:
                                print(f"PARAM {pid} COULD NOT BE FOUND in {fid} or its interactants... ERR")

                                # extend field with return key of method since
                                fattrs["keys"].append(pid)
                                fattrs["values"].append(None)

                                param_collector.append(
                                    self.get_db_index(
                                        module["module_index"],
                                        field_index,
                                        fattrs["keys"].index(pid),
                                    )
                                )
                                print(f"Added ghost param {pid} to {fid}")
                                field_eq_param_struct.append(param_collector)

                        print(f"finished field_eq_param_struct for {fid}:")
                        print("field_eq_param_struct", len(field_eq_param_struct))

                        expand_field_eq_variation_struct = expand_structure(
                            struct=field_eq_param_struct
                        )

                        print("expand DB_TO_METHOD_EDGES")

                        # flat expand edge struct
                        for item in expand_field_eq_variation_struct:
                            db_to_method["DB_TO_METHOD_EDGES"][eq_idx].extend(item)
                            # todo calc just / len(method_param) to sort them

                        print("expand_field_eq_variation_struct", len(expand_field_eq_variation_struct))
                        EQ_AMOUNT_VRIATIONS += len(field_eq_param_struct)

                    db_to_method[
                        "DB_CTL_VARIATION_LEN_PER_EQUATION"
                    ][m_idx].append(EQ_AMOUNT_VRIATIONS)

            flatten_variations = []
            for item in db_to_method["DB_TO_METHOD_EDGES"]:
                flatten_variations.extend(item)
            db_to_method["DB_TO_METHOD_EDGES"] = flatten_variations

            # extend all mdules
            flatten_amount_variations = []
            for item in db_to_method["DB_CTL_VARIATION_LEN_PER_EQUATION"]:
                flatten_amount_variations.extend(item)
            db_to_method["DB_CTL_VARIATION_LEN_PER_EQUATION"] = flatten_amount_variations
            print("set_edge_db_to_method... done")

        except Exception as e:
            print(f"Err set_edge_db_to_method: {e}")
        print("db_to_method", db_to_method)
        return db_to_method


    def set_iterator_from_humans(self, ):

        iterator = {
            "MODULES": [],  # Anzahl Gleichungen pro Modul [n1, n2, ...]
            "FIELD_VARIATIONS": [],
            "FIELDS": [],  # Anzahl Felder pro Modul [f1, f2, ...]
        }

        modules = self.g.get_nodes(filter_key="type", filter_value="MODULE")
        ghost_fields = self.g.get_neighbor_list_rel(trgt_rel="has_field", node="GHOST_MODULE")

        try:
            for m_idx, (mid, module) in enumerate(modules):
                if "GHOST" in mid.upper():
                    continue

                # A. Felder des Moduls sammeln
                fields = self.g.get_neighbor_list_rel(node=mid, trgt_rel="has_field", as_dict=True)
                len_fields = len(fields)
                iterator["FIELDS"].append(len_fields)

                # B. Methoden (Gleichungen) des Moduls sammeln
                methods = self.g.get_neighbor_list_rel(trgt_rel="has_method", node=mid, as_dict=True)
                iterator["MODULES"].append(len(methods))

                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    # Parameter-Definitionen der Gleichung
                    params = eqattrs.get("params", [])
                    if isinstance(params, str): params = json.loads(params)
                    params_origin = eqattrs.get("origin", None) or ["" for _ in range(len(params))]

                    # Pro Gleichung: Wie viele Parameter erwartet sie insgesamt?

                    for fid, fattrs in fields.items():
                        # Vorbereitung der Keys des aktuellen Feldes
                        keys = fattrs.get("keys", [])
                        if isinstance(keys, str): keys = json.loads(keys)

                        field_index = fattrs.get("field_index")
                        fneighbors = self.g.get_neighbor_list_rel(node=fid, trgt_rel="has_finteractant", as_dict=True)

                        # Struktur für die Parameter-Zuweisung dieses Feldes
                        field_eq_param_struct = []

                        for pidx, pid in enumerate(params):
                            is_prefixed = pid.endswith("_")
                            clean_pid = pid[:-1] if is_prefixed else pid
                            collected = False
                            param_collector = []

                            # 1. Check: Gehört der Parameter zum Feld selbst?
                            if clean_pid in keys and not is_prefixed and params_origin[pidx] not in ["neighbor", "interactant"]:
                                pindex = keys.index(clean_pid)

                                field_eq_param_struct.append(
                                    self.get_db_index(
                                        module["module_index"],
                                        field_index,
                                        pindex,
                                    )
                                )
                                collected = True
                            else:
                                # 2. Check: Ist es ein Interactant (Neighbor)?
                                for finid, fiattrs in fneighbors.items():
                                    ikeys = fiattrs.get("keys", [])
                                    if isinstance(ikeys, str): ikeys = json.loads(ikeys)

                                    if clean_pid in ikeys:
                                        pindex = ikeys.index(clean_pid)
                                        nfield_index = fiattrs.get("field_index")
                                        pmod = self.g.get_node(nid=fiattrs["module_id"])
                                        param_collector.append(
                                            self.get_db_index(
                                                pmod["module_index"],
                                                nfield_index,
                                                pindex,
                                            ))
                                        collected = True

                                # 3. Check: Ghost-Felder (Globaler Fallback)
                                if not collected:
                                    for gfid, gfattrs in ghost_fields:
                                        gikeys = gfattrs.get("keys", [])
                                        gfield_index = gfattrs.get("field_index", [])
                                        pmod = self.g.get_node(nid="GHOST_MODULE")

                                        if isinstance(gikeys, str): gikeys = json.loads(gikeys)
                                        if clean_pid in gikeys:
                                            pindex = gikeys.index(clean_pid)
                                            param_collector.append(
                                                self.get_db_index(
                                                    pmod["module_index"],
                                                    gfield_index,
                                                    pindex,
                                                )
                                            )
                                            collected = True

                                field_eq_param_struct.append(param_collector if collected else -1)

                        # Upscaling: Berechne Variationen (Kombinatorik der Interaktions-Parameter)
                        # Identifiziert die maximale Tiefe (Anzahl der Interaktanten) für dieses Feld
                        list_params = [x for x in field_eq_param_struct if isinstance(x, list)]
                        num_variations = len(max(list_params, key=len)) if list_params else 1
                        iterator["FIELD_VARIATIONS"].append(num_variations)

            #print("set_iterator_from_humans: GPU Skeleton successfully compiled.")
            return iterator

        except Exception as e:
            print(f"Error in set_iterator_from_humans: {e}")
            raise e


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
                    print("set_edge_db_to_method... len methods 0")
                    continue

                print("set_edge_db_to_method meq", type(methods))

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

            #nit with size max_index + 1
            modules_struct = []
            print("modules_struct initialized size:", len(modules_struct))

            for i, (mid, m) in enumerate(modules):
                if include_ghost_mod is None or include_ghost_mod is False:
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
                keys = list(fields.keys())
                print(f"get_empty_field_structure fields for {mid}")

                field_struct = []
                for _ in range(len(keys)):
                    field_struct.append([])

                print(f">>>{mid} field_struct:", keys, len(keys))

                # SET EMPTY FIELDS STRUCT AT MODULE INDEX
                modules_struct.append(field_struct)
            print("get_empty_field_structure... done")
            return modules_struct
        except Exception as e:
            print("Err get_empty_field struct:", e)



    def get_db_index(self, mod_idx, field_idx, param_in_field_idx):
        # get_db_index
        return (
            mod_idx,
            field_idx,
            param_in_field_idx,
        )



    def get_empty_method_structure(self, set_zero=True, ):
        methods: list = self.g.get_nodes(
            filter_key="type",
            filter_value="METHOD",
        )
        try:
            print("modules_struct... done")
            return [
                0 if set_zero is True else []
                for _ in methods
            ]
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

    guard.main(
        env_id="env_7c87bb26138a427eb93cab27d0f5429f",
        env_data=payload["data"]["config"]["env_7c87bb26138a427eb93cab27d0f5429f"],
    )




















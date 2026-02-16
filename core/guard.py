import base64
import pprint

from itertools import product

import numpy as np
from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin
import networkx as nx
from utils.graph.local_graph_utils import GUtils
from qf_utils.qf_utils import QFUtils

from core.qbrain_manager import get_qbrain_table_manager

from core.module_manager.mcreator import ModuleCreator
from qf_utils.all_subs import ALL_SUBS
from utils._np.expand_array import expand_structure
from utils.get_shape import extract_complex, get_shape
from utils.xtract_trailing_numbers import extract_trailing_numbers
from vertex_trainer.manager import VertexTrainerManager
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

    """

    def __init__(
        self,
        qfu,
        g,
        user_id,
        field_manager=None,
        method_manager=None,
        injection_manager=None,
        env_manager=None,
        module_db_manager=None,
    ):
        print("Initializing Guard...")

        PatternMaster.__init__(
            self,
            g,
        )
        self.user_id = user_id
        self.field_manager = field_manager
        self.method_manager = method_manager
        self.injection_manager = injection_manager
        self.env_manager = env_manager
        self.module_db_manager=module_db_manager

        self.deployment_handler = DeploymentHandler(
            user_id
        )

        self.qb = get_qbrain_table_manager()
        print("DEBUG: QBrainTableManager initialized")

        self.world_cfg=None
        print("DEBUG: ArtifactAdmin initialized")
        self.artifact_admin = ArtifactAdmin()
        self.trainer = VertexTrainerManager()
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

        self.prev_state = None
        self.model_params = None
        self.fnished_modules = False

        self.fields = []
        print("Guard Initialized!")


    def create_nodes(self, env_id, env_data):
        print("create_nodes...")
        pprint.pp(env_data)

        # RESET G
        self.g.G = nx.Graph()

        # 1. Collect IDs
        module_ids = set()
        field_ids = set()

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

        #############
        # INJECTIONS
        injections = env_data.get("injections", {})
        inj_ids = self.handle_injections(injections)

        print("create_nodes... done")
        return module_ids, field_ids, inj_ids



    def handle_injections(self, injections):
        print("handle_injections...", injections)
        inj_ids = set()
        for field_id, data in injections.items():
            for pos, inj_id in data.items():
                inj_ids.add(inj_id)

                inj_id = f"{pos}__{inj_id}"
                print("ADD INJECTION", inj_id)
                self.g.add_node(
                    {
                        "nid": inj_id,
                        "type": "INJECTION",
                    }
                )
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
        print("handle_injections...", inj_ids)
        return inj_ids

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
        #self.g.print_status_G()
        print("create_edges... done")


    def check_create_ghost_mod(self, env_id):
        ghost_mod_id = f"GHOST_MODULE"
        print(f"Creating Ghost Module: {ghost_mod_id}")

        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
            just_id=True,
        )
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

        #add fallack None values in case params couldnt be found
        keys = list(self.qfu.create_env().keys())

        # FETCH ENV CONSTANTS
        env_constants = self.qb.row_from_id(keys, table="params")

        # todo save params fpr env altimes in row  -> save this trash here
        # VALUE # KEYS # AXIS
        values = [v["value"] for v in env_constants]
        keys = [v["id"] for v in env_constants]
        axis_def = [v["axis_def"] for v in env_constants]

        values.append(None)
        keys.append("None")
        axis_def.append(None)

        # FETCH ENV
        res = self.qb.row_from_id(
            [env_id],
            table=self.env_manager.TABLE_ID
        )

        # GET DATA
        res = res[0]
        self.g.add_node(
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

        env_data = dict(
            nid=env_id,
            type="FIELD",
            sub_type="ENV",
            keys=keys,
            values=values,
            axis_def=axis_def,
            field_index=len(fields),
        )
        print("ADD ENV FIELD")
        self.g.add_node(env_data)

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
            #print("fattrs", fattrs)
            interactant_fields:list[str] or str = fattrs["interactant_fields"]
            #print(f"{fid} interactant fields: {len(interactant_fields)}")

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

                #print("fetched_modules")

                 # ADD MISSING FIELD NODES
                for mfid, mfattrs in fetched_modules.items():
                    values = json.loads(mfattrs.get("values"))
                    keys = json.loads(mfattrs.get("keys"))

                    #print(f"ADD FINTERACTANT NODE {mfid}")
                    self.g.add_node(
                        dict(
                            nid=mfid,
                            type="FIELD",
                            keys=keys,
                            value=values,
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
                            #print("FIELD INTERACTANT -> MODULE")
                            self.g.add_edge(
                                mfattrs.get("module_id"),
                                fid,
                                attrs={
                                    "rel": "has_field",
                                    "src_layer": "MODULE",
                                    "trgt_layer": "FIELD",
                                }
                            )
                        else:
                           # print("MODULE WAS NOT CHOSSED BY USER -> GHOST FIELD")
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
        print("main...")
        #pprint.pp(env_data)
        self.data_handler(env_id, env_data)
        components = self.converter(env_id)
        self.handle_deployment(env_id, components)
        print("main... done")

    def handle_deployment(self, env_id, components):
        print(f"\n[START] handle_deployment für Env: {env_id}")
        try:
            # 1. Config Erstellung
            world_cfg = self.create_vm_cfgs(env_id)
            print(f"  -> world_cfg erstellt:", type(world_cfg))
            pprint.pp(world_cfg)

            # 2. DB Update
            self.module_db_manager.qb.update_env_pattern(
                env_id=env_id,
                pattern_data=components,
                user_id=self.user_id
            )

            # 3. Env Variables
            container_env:dict = self.deployment_handler.env_creator.create_env_variables(
                env_id=env_id,
                cfg=world_cfg
            )
            print("  -> Container-Umgebungsvariablen generiert")

            # 4. Payload Generierung
            latest_image = self.artifact_admin.get_latest_image()



            # LAUNCH TRAIING JOB VAIs
            self.trainer.create_custom_job(
                display_name=env_id,
                container_image_uri=latest_image,
                container_envs=container_env,
            )

            # 6. Status Update
            from core.managers_context import get_user_manager
            get_user_manager().qb.set_item(
                "envs",
                {"status": "IN_PROGRESS"},
                keys={"id": env_id, "user_id": self.user_id}
            )
            print("  -> Status in DB auf 'IN_PROGRESS' gesetzt")

            print(f"  -> Modus: TESTING (Speichere lokal)")
            path = r"C:\Users\bestb\PycharmProjects\BestBrain\test_out.json"
            with open(path, "w") as f:
                f.write(json.dumps(components, indent=4))
            print(f"  -> Datei erfolgreich geschrieben unter: {path}")

        except Exception as e:
            print(f"  [!!!] ERROR in handle_deployment: {e}")
            import traceback
            traceback.print_exc()
            try:
                from core.managers_context import get_user_manager
                get_user_manager().qb.set_item(
                    "envs",
                    {"status": "FAILED"},
                    keys={"id": env_id, "user_id": self.user_id}
                )
            except Exception:
                pass

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
                            import json
                            fattrs["interactant_fields"] = json.loads(fattrs["interactant_fields"])
                        except Exception as e:
                            print(f"Error parsing interactant_fields for {fid}: {e}")


                    # create complex map for tbhe db indexing (exclude nid/type from fattrs to avoid duplicate kwargs)
                    attrs = {"nid": fid, "type": "FIELD"}
                    for k, v in fattrs.items():
                        if k not in ("nid", "type"):
                            attrs[k] = v
                    self.g.add_node(attrs)
                else:
                    self.g.add_node({"nid": fid, "type": "FIELD"})

        # GET INJ BQ
        if inj_ids:
            print("Fetching injections from BQ...")

            res = self.qb.row_from_id(
                list(inj_ids),
                table=self.injection_manager.table
            )

            fetched_injections = {item["id"]: item for item in res}

            for inid, inj_data in fetched_injections.items():
                print(f"inj_data: {inid} : {inj_data}")

                for k, v in self.g.G.nodes(data=True):
                    if v.get("type") == "INJECTION" and inid in k:
                        if inj_data:
                            attrs = {"nid": k, "type": "INJECTION"}
                            for pk, pv in inj_data.items():
                                if pk not in ("nid", "type"):
                                    attrs[pk] = pv
                            self.g.add_node(attrs)
            print(f"Fetched injections: {len(fetched_injections)} ")

        modules_config = env_data.get("modules", {})
        #print(f"Processing {len(modules_config)} modules from config...")

        for mod_id, mod_config in modules_config.items():
            mod_id = mod_id.upper()
            #print(f"Processing module: {mod_id}")

            mod_data = fetched_modules.get(mod_id)

            if mod_data:
                self.g.add_node(dict(
                    nid=mod_id,
                    type="MODULE",
                    **{k:v for k,v in mod_data.items() if k not in self.g.get_node(nid=mod_id)}
                ))

            else:
                self.g.add_node({"nid": mod_id, "type": "MODULE"})

            fields_config = mod_config.get("fields", {})
            #print(f"Processing {len(fields_config)} fields for module {mod_id}...")

            for field_id, field_config in fields_config.items():
                field_id = field_id.upper()
                #print(f"Processing field: {field_id}")
                field_data = fetched_fields.get(field_id)
                if field_data:
                    import json
                    values = json.loads(field_data.get("values"))
                    keys = json.loads(field_data.get("keys"))

                    if "interactant_fields" in field_data and isinstance(
                            field_data["interactant_fields"],
                            str
                    ):
                        try:
                            field_data["interactant_fields"] = json.loads(field_data["interactant_fields"])
                        except Exception as e:
                            print(f"Error parsing field data interactant_fields for {field_id}: {e}")

                    attrs = {"nid": field_id, "type": "FIELD", "keys": keys, "value": values}
                    for k, v in field_data.items():
                        if k not in ("nid", "type", "values", "keys"):
                            attrs[k] = v
                    self.g.add_node(attrs)

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
                    self.g.add_node({
                        "nid": field_id,
                        "type": "FIELD",
                    })



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

        # include gm
        modules: list = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        self.DB = self.create_db(
            modules,
        )

        # create iterator last
        iterators = self.set_iterator_from_humans()

        db_to_method_struct:dict = self.set_edge_db_to_method(modules, env_id)

        method_struct:dict = self.method_layer(modules)

        method_to_db = self.set_edge_method_to_db()

        injection_patterns = self.set_inj_pattern(env_node)

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
        if not vm_payload or not isinstance(vm_payload, dict):
            print("  [WARN] deploy_vms: empty or invalid vm_payload, skipping")
            return
        for key, cfg in vm_payload.items():
            if not isinstance(cfg, dict):
                print(f"  [WARN] deploy_vms: skipping invalid cfg for key={key}")
                continue
            print("deploy vm from", cfg)


            try:
                self.deployment_handler.create_instance(
                    **cfg
                )
            except Exception as e:
                print(f"  [!!!] deploy_vms failed for {key}: {e}")
                import traceback
                traceback.print_exc()
                raise
        print("Deployment Finished!")


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

        return world_cfg




    def set_inj_pattern(
            self,
            env_attrs,
            trgt_keys=["energy", "j_nu", "vev"], # injectable parameters
    ):
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

                midx = mattrs["module_index"]
                print(f"MODULE index for {mid}: {midx}")

                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True
                )

                for fid, fattrs in fields.items():
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

                                print(f"set param pathway db from mod {midx} -> field {fattrs['nid']}({fi})")
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

                self.g.add_node(
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


    def create_db(self, modules):
        print("[create_db] start")
        import json
        # --- initialize base structures ---
        db = self.get_empty_field_structure()
        axis = self.get_empty_field_structure()
        shapes = self.get_empty_field_structure()
        item_len_collection = self.get_empty_field_structure()
        param_len_collection = self.get_empty_field_structure()
        field_ids = self.get_empty_field_structure()
        db_keys = self.get_empty_field_structure()

        DB = {
            "DB": [],
            "AXIS": [],
            "DB_SHAPE": [],
            "AMOUNT_PARAMS_PER_FIELD": [],
            "DB_PARAM_CONTROLLER": [],
            "DB_KEYS": [],
            "FIELD_KEYS": []
        }


        try:
            for mid, m in modules:
                m_idx = m["module_index"]

                print(f"[create_db] module={mid}, idx={m_idx}")

                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True
                )

                for fid, fattrs in fields.items():
                    fidx = fattrs["field_index"]

                    #print(f"[field] apply pattern -> module={m_idx}, field={fidx}")

                    # axis definition
                    xdef = fattrs.get("axis_def", [])
                    if isinstance(xdef, str):
                        xdef = json.loads(xdef)
                    #print(f"{fid} xdef", xdef)

                    # values
                    vals = fattrs.get("values", [])
                    if isinstance(vals, str):
                        vals = json.loads(vals)

                    # keys
                    keys = fattrs.get("keys", [])
                    if isinstance(keys, str):
                        keys = json.loads(keys)

                    #print(f"{fid} keys | vals", len(keys), "|", len(vals))
                    shape_struct = []
                    collected_values = []
                    vals_item_lens = []


                    # --- process values ---
                    for vi, (v, k, xd) in enumerate(zip(vals, keys, xdef)):
                        shape = get_shape(v)
                        #print(f"shape for {k}:{v}:{shape}")
                        shape_struct.append(shape)

                        # flatten any shape -> 1d
                        sval_collector = []

                        extract_complex(v, sval_collector)
                        vals_item_lens.append(
                            len(sval_collector) # add len 1d numbers
                        )
                        collected_values.extend(sval_collector)

                    # --- assign per module / field ---
                    db[m_idx][fidx] = collected_values
                    axis[m_idx][fidx] = xdef
                    shapes[m_idx][fidx] = shape_struct
                    item_len_collection[m_idx][fidx] = vals_item_lens
                    param_len_collection[m_idx][fidx] = len(vals)
                    db_keys[m_idx][fidx] = keys
                    field_ids[m_idx][fidx] = fid


            # len is correctly set
            for i, (m_db, m_axis, m_shape, m_len, plen_item, db_key_struct) in enumerate(
                    zip(db, axis, shapes, item_len_collection, param_len_collection, db_keys)):

                for j, (f_db, f_axis, f_shape, f_len, plen_field, db_key) in enumerate(
                        zip(m_db, m_axis, m_shape, m_len, plen_item, db_key_struct)):

                    DB["DB"].extend(f_db)
                    DB["AXIS"].extend(f_axis)
                    DB["DB_SHAPE"].extend(f_shape)

                    # int len each single param in db unscaled (e.g. 3 for 000)
                    DB["DB_PARAM_CONTROLLER"].extend(f_len)

                    # AMOUNT_PARAMS_PER_FIELD
                    # int len param / f
                    DB["AMOUNT_PARAMS_PER_FIELD"].append(plen_field)
                    DB["DB_KEYS"].extend(db_key)

            DB["DB"] = base64.b64encode(
                np.array(DB["DB"], dtype=np.complex64).tobytes()
            ).decode("utf-8")

            print("[create_db] done")
        except Exception as e:
            print("[create_db][ERROR]", e)
        return DB



    def handle_methods(self, module_ids:list[str]):
        print("handle_methods...")
        import json
        method_idx = 0
        for mid in module_ids:
            mod_node = self.g.get_node(nid=mid)
            mmethods = mod_node["methods"]

            if isinstance(mmethods, str):
                mmethods = json.loads(mmethods)

            #print("mod_node methods", mmethods)

            methods = self.qb.row_from_id(
                nid=mmethods,
                table=self.method_manager.METHODS_TABLE,
            )
            #print("methods", methods)

            methods = {
                item["id"]: item
                for item in methods
            }
            #print("methods1", methods)

            for k, (method_id, method_data) in enumerate(methods.items()):
                self.g.add_node({
                    "nid": method_id,
                    "type": "METHOD",
                    "method_idx": method_idx,
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
                method_idx += 0
        print("handle_methods... done")



    def method_layer(self, modules):
        print("method_layer... ")

        mod_len_exclude_ghost = len(modules) -1
        #print("method_layer mod_len_exclude_ghost", mod_len_exclude_ghost)

        method_struct = {
            "METHOD_PARAM_LEN_CTLR": [[] for _ in range(mod_len_exclude_ghost)],
            "METHODS": [[] for _ in range(mod_len_exclude_ghost)],
            "METHODS_PER_MOD_LEN_CTLR": [0 for _ in range(mod_len_exclude_ghost)],
        }
        mnames = [[] for _ in range(mod_len_exclude_ghost)]
        try:
            #print("method_layer compilation...")
            for mid, module in modules:
                # ghost does not have equation
                if "GHOST" in mid.upper(): continue
                #print("method_layer... working", mid)

                midx:int = module.get("module_index")
                #print("midx", midx)

                #print("method_layer... module_index")

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )
                mids = list(methods.keys())
                mlen:int = len(mids)
                if not mlen:
                    print("method_layer... len methods", mlen)
                    continue

                # len methods per module
                method_struct["METHODS_PER_MOD_LEN_CTLR"][midx] = mlen

                mnames[midx].extend(mids)

                # Iterate Equations
                for eqid, eqattrs in methods.items():
                    # PARAMS from METHOD
                    params = eqattrs.get("params", [])

                    if isinstance(params, str):
                        params = json.loads(params)

                    method_struct["METHOD_PARAM_LEN_CTLR"][midx].append(
                        len(params)
                    )

                    method_struct["METHODS"][midx].append(
                        eqattrs.get("jax_code", eqattrs.get("code"))
                    )

            # flatten
            flattened_methods = []
            for i, sublist in enumerate(method_struct["METHODS"]):
                """
                for j, mid in enumerate(mnames[i]):
                    print(f"mod {i} method {j}: {mid}")
                """
                flattened_methods.extend(sublist)

            method_struct["METHODS"] = flattened_methods

            flatten_ctlr = []
            for sublist in method_struct["METHOD_PARAM_LEN_CTLR"]:
                flatten_ctlr.extend(sublist)
            method_struct["METHOD_PARAM_LEN_CTLR"] = flatten_ctlr

        except Exception as e:
            print("Err method_layer", e)
        #pprint.pp(method_struct)
        print(f"method_layer... done")
        return method_struct


    def set_edge_method_to_db(self):
        # each eqs fields has different return key (IMPORTANT: sum variation results)
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE"
        )

        return_key_map = [
            []
            for _ in range(len(modules)-1)
        ]

        try:
            print("start compilation...")
            for mid, module in modules:
                if "GHOST" in mid.upper(): continue
                print("set_edge_method_to_db... working", mid)

                m_idx = module.get("module_index")

                # GET MODULES METHODS
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                if not len(list(methods.keys())):
                    print("set_edge_method_to_db... len methods 0")
                    continue

                #print("set_edge_db_to_method meq", type(methods))

                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                for eqid, eqattrs in methods.items():
                    # print("eqattrs", eqattrs)
                    #print(f"Equation: {eqid}")

                    for fid, fattrs in fields.items():
                        keys: list[str] or str = fattrs.get("keys", [])

                        return_key = eqattrs.get("return_key")
                        field_index = fattrs["field_index"]
                        rindex = keys.index(return_key)

                        return_key_map[m_idx].append(
                            self.get_db_index(
                                m_idx,
                                field_index,
                                rindex,
                            )
                        )
            # flatte
            flatten_rtk_map = []
            for module_items in return_key_map:
                flatten_rtk_map.extend(module_items)
            return flatten_rtk_map
        except Exception as e:
            print("Err set_edge_method_to_db", e)
            pass



    def set_edge_db_to_method(self, modules, env_id):
        print("set_edge_db_to_method...")
        # todo for field index includ again
        mlen = len(modules)-1
        import json
        # db out gnn
        db_to_method = {
            "DB_TO_METHOD_EDGES": [[] for _ in range(mlen)],
            "DB_CTL_VARIATION_LEN_PER_EQUATION": [[] for _ in range(mlen)],
            "DB_CTL_VARIATION_LEN_PER_FIELD": [[] for _ in range(mlen)],
            "LEN_FEATURES_PER_EQ": self.get_empty_method_structure(set_zero=False),
            "VARIATION_KEYS": [[] for _ in range(mlen)]
        }

        ghost_fields = self.g.get_neighbor_list_rel(
            trgt_rel="has_field",
            node="GHOST_MODULE",
        )

        try:
            print("start compilation...")
            for mid, module in modules:
                if "GHOST" in mid.upper(): continue
                m_idx = module["module_index"]
                #print("set_edge_db_to_method... working", mid)

                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                if not methods:
                    #print("set_edge_db_to_method... len methods 0")
                    continue

                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )
                if not fields:
                    raise Exception(f"Err: no fields found for module {mid}")

                for eq_idx, (eqid, eqattrs) in enumerate(methods.items()):
                    EQ_AMOUNT_VRIATIONS = 0

                    params = eqattrs.get("params", [])
                    if isinstance(params, str): params = json.loads(params)

                    params_origin = eqattrs.get("origin", [])
                    if isinstance(params_origin, str): params_origin = json.loads(params_origin)
                    if not params_origin: params_origin = [""] * len(params)

                    if params_origin is None:
                        params_origin = [
                            ""
                            for _ in range(len(params))
                        ]
                    #print("methods params_origin", params_origin)

                    #
                    if isinstance(params, str):
                        params = json.loads(params)

                    for fid, fattrs in fields.items():
                        field_index = fattrs["field_index"]
                        # Space to save all variations for all inteactions for all equations
                        field_eq_param_struct = []

                        if isinstance(fattrs["keys"], str):
                            fattrs["keys"] = json.loads(fattrs["keys"])

                        keys: list[str] or str = fattrs.get("keys", [])

                        if isinstance(keys, str):
                            keys = json.loads(keys)

                        fneighbors = self.g.get_neighbor_list_rel(
                            node=fid,
                            trgt_rel="has_finteractant",
                            as_dict=True
                        )

                        # > 500 ?

                        # LOOP EQ-PARAMS
                        for pidx, pid in enumerate(params):
                            collected = False

                            param_collector = []
                            param_origin_key_collector = []
                            #print("work pid", pid)

                            # Field's own param
                            is_prefixed = pid.endswith("_")
                            is_self_prefixed = pid.startswith("_")
                            id_prev_val = pid.startswith("prev")
                            EXCLUDED_ORIGINS = ["neighbor", "interactant"]

                            if (
                                pid in keys and (not is_prefixed or is_self_prefixed) and params_origin[pidx] not in EXCLUDED_ORIGINS
                            ):
                                print(f"{pid} in {fid}")

                                time_dim = None

                                # RM start "_"
                                if id_prev_val:
                                    #print("prev etected:", pid)
                                    # _prev must be in keys!!!
                                    pid = pid.replace("prev_","").strip()
                                    time_dim = 1

                                elif is_self_prefixed:
                                    pid = pid[:-1]

                                if time_dim is None:
                                    time_dim = 0

                                pindex = keys.index(pid)

                                result = self.get_db_index(
                                    m_idx,
                                    field_index,
                                    pindex,
                                    time_dim,
                                )

                                field_eq_param_struct.append(
                                    result
                                )

                                collected = True
                                continue
                            else:
                                pid = pid[:-1] if is_prefixed else pid

                                for finid, fiattrs in fneighbors.items():
                                    ikeys = fiattrs.get("keys")

                                    if isinstance(ikeys, str):
                                        ikeys = json.loads(ikeys)

                                    # param key in interactant field?
                                    if pid in ikeys:
                                        fmod = self.g.get_node(nid=fiattrs.get("module_id"))

                                        # collect maps for all interactants
                                        #print("interactant pid", pid, o)

                                        nfield_index = fiattrs["field_index"]

                                        pindex = ikeys.index(pid)
                                        #print("interactant pindex", pindex, o)

                                        param_collector.append(
                                            self.get_db_index(
                                                fmod["module_index"],
                                                nfield_index,
                                                pindex,
                                            )
                                        )
                                        # collect field interaction keys
                                        param_origin_key_collector.append(finid)
                                        collected = True

                                #print(f"param {pid} is not in interactant -> check GHOST FIELDS")
                                for gfid, gfattrs in ghost_fields:
                                    gikeys = gfattrs.get("keys")

                                    if isinstance(gikeys, str):
                                        #print("convert ikeys", gikeys)
                                        gikeys = json.loads(gikeys)

                                    if pid in gikeys:
                                        gfield_index = gfattrs["field_index"]

                                        pindex = gikeys.index(pid)
                                        #print("interactant pindex", pindex)
                                        #print(f"{pid} found in gfid ({gfield_index}) (pindex {pindex})", gfield_index)

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

                                if collected is False:
                                    #print(f"PARAM {pid} COULD NOT BE FOUND in {fid} or its interactants... ERR")
                                    gmod = self.g.get_node(nid="GHOST_MODULE")
                                    env_field = self.g.get_node(nid=env_id)
                                    pindex = env_field["keys"].index("None")
                                    param_collector.append(
                                        self.get_db_index(
                                            gmod["module_index"],
                                            env_field["field_index"],
                                            pindex,
                                        )
                                    )

                            field_eq_param_struct.append(param_collector)

                        #print(f"finished field_eq_param_struct for {fid}:")
                        #print(f"field_eq_param_struct {eqid} {fid}", len(field_eq_param_struct))
                        #pprint.pp(field_eq_param_struct)

                        expand_field_eq_variation_struct = expand_structure(
                            struct=field_eq_param_struct
                        )

                        # fehelr: eq_idx ist relativ an modul bound.
                        # todo: untertiele method nach modul index.
                        #  mach heir das gleiche und flatte im Anschlusse

                        # extend variation single eq
                        for item in expand_field_eq_variation_struct:
                            db_to_method["DB_TO_METHOD_EDGES"][m_idx].extend(item)
                            #print(f"module {mid} ({m_idx}) expand_field_eq_variation_struct item {eqid}", item)
                            # todo calc just / len(method_param) to sort them

                        #print("expand_field_eq_variation_struct", len(expand_field_eq_variation_struct))

                        field_variations_eq = len(expand_field_eq_variation_struct)
                        EQ_AMOUNT_VRIATIONS += field_variations_eq

                        # add len field var to emthod struct
                        db_to_method[
                            "LEN_FEATURES_PER_EQ"
                        ][eq_idx].append(field_variations_eq)

                        db_to_method[
                            "DB_CTL_VARIATION_LEN_PER_FIELD"
                        ][m_idx].append(
                            len(expand_field_eq_variation_struct)
                        )

                    # todo combine amount variation / field -&- shape / field / eq:
                    #  DB_CTL_VARIATION_LEN_PER_FIELD, LEN_FEATURES_PER_EQ

                    db_to_method[
                        "DB_CTL_VARIATION_LEN_PER_EQUATION"
                    ][m_idx].append(EQ_AMOUNT_VRIATIONS)

            flatten_variations = []
            for i, item in enumerate(db_to_method["DB_TO_METHOD_EDGES"]):
                flatten_variations.extend(item)
            db_to_method["DB_TO_METHOD_EDGES"] = flatten_variations

            flatten_amount_variations = []
            for item in db_to_method["DB_CTL_VARIATION_LEN_PER_EQUATION"]:
                flatten_amount_variations.extend(item)
            db_to_method["DB_CTL_VARIATION_LEN_PER_EQUATION"] = flatten_amount_variations

            # for sepparation for the sum process
            flatten_amount_variations_per_field = []
            for item in db_to_method["DB_CTL_VARIATION_LEN_PER_FIELD"]:
                flatten_amount_variations_per_field.extend(item)
            db_to_method["DB_CTL_VARIATION_LEN_PER_FIELD"] = flatten_amount_variations_per_field
            print("set_edge_db_to_method... done")
        except Exception as e:
            print(f"Err set_edge_db_to_method: {e}")
            raise
        return db_to_method



    def set_iterator_from_humans(self):
        iterator = {
            "MODULES": [],
            "FIELDS": [],
        }
        import json
        modules = self.g.get_nodes(filter_key="type", filter_value="MODULE")
        ghost_fields = self.g.get_neighbor_list_rel(trgt_rel="has_field", node="GHOST_MODULE")

        try:
            for i, (mid, module) in enumerate(modules):
                m_idx = module.get("module_index")
                # A. FeDB_PARAM_CONTROLLERlder des Moduls sammeln
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

                        field_index = fattrs["field_index"]
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
                                        nfield_index = fiattrs["field_index"]
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
                                        gfield_index = gfattrs["field_index"]
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
        import json
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
        import json
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
        import json
        modules_struct = []
        try:

            #nit with size max_index + 1
            print("modules_struct initialized size:", len(modules_struct))

            for mid, m in modules:
                if include_ghost_mod is False:
                    if "GHOST" in mid.upper(): continue


                # get module fields
                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                keys = list(fields.keys())
                #print(f"get_empty_field_structure fields for {mid}")

                field_struct = []
                for _ in range(len(keys)):
                    field_struct.append([])

                #print(f">>>{mid} field_struct:", keys, len(keys))

                # SET EMPTY FIELDS STRUCT AT MODULE INDEX
                modules_struct.append(field_struct)

        except Exception as e:
            print("Err get_empty_field struct:", e)
        print("get_empty_field_structure... done")
        return modules_struct

    def get_db_index(self, mod_idx, field_idx, param_in_field_idx, time_dim=0):
        return (
            time_dim, # the t-dim to choose the item from
            mod_idx,
            field_idx,
            param_in_field_idx,
        )

    def get_empty_method_structure(self, set_zero=True, ):
        modules: list = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )
        mlen_excl_ghost = len(modules)-1
        method_struct = [
            [] if set_zero is False else 0
            for _ in range(mlen_excl_ghost)
        ]
        try:

            # nit with size max_index + 1
            modules_struct = []
            print("modules_struct initialized size:", len(modules_struct))

            for mid, m in modules:
                if "GHOST" in mid.upper(): continue

                module_index = m["module_index"]

                method_nodes = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )

                method_struct[module_index] = [
                    [] if set_zero is False else 0
                    for _ in range(len(method_nodes))
                ]

            flattened_methods = []
            for i in method_struct:
                flattened_methods.extend(i)

            return flattened_methods
        except Exception as e:
            print("Err get_empty_field struct:", e)


if __name__ == "__main__":
    import asyncio
    import json
    import websockets

    # Define payload (same structure as relay expects)
    payload = {'type': 'START_SIM', 'data': {'config': {'env_7c87bb26138a427eb93cab27d0f5429f': {'modules': {'GAUGE': {'fields': {'photon': {'injections': {'[4,4,4]': 'hi'}}, 'w_plus': {'injections': {}}, 'w_minus': {'injections': {}}, 'z_boson': {'injections': {}}, 'gluon_0': {'injections': {}}, 'gluon_1': {'injections': {}}, 'gluon_2': {'injections': {}}, 'gluon_3': {'injections': {}}, 'gluon_4': {'injections': {}}, 'gluon_5': {'injections': {}}, 'gluon_6': {'injections': {}}, 'gluon_7': {'injections': {}}}}, 'HIGGS': {'fields': {'phi': {'injections': {}}}}, 'FERMION': {'fields': {'electron': {'injections': {}}, 'muon': {'injections': {}}, 'tau': {'injections': {}}, 'electron_neutrino': {'injections': {}}, 'muon_neutrino': {'injections': {}}, 'tau_neutrino': {'injections': {}}, 'up_quark_0': {'injections': {}}, 'up_quark_1': {'injections': {}}, 'up_quark_2': {'injections': {}}, 'down_quark_0': {'injections': {}}, 'down_quark_1': {'injections': {}}, 'down_quark_2': {'injections': {}}, 'charm_quark_0': {'injections': {}}, 'charm_quark_1': {'injections': {}}, 'charm_quark_2': {'injections': {}}, 'strange_quark_0': {'injections': {}}, 'strange_quark_1': {'injections': {}}, 'strange_quark_2': {'injections': {}}, 'top_quark_0': {'injections': {}}, 'top_quark_1': {'injections': {}}, 'top_quark_2': {'injections': {}}, 'bottom_quark_0': {'injections': {}}, 'bottom_quark_1': {'injections': {}}, 'bottom_quark_2': {'injections': {}}}}}}}}, 'auth': {'session_id': 339617269692277, 'user_id': '72b74d5214564004a3a86f441a4a112f'}, 'timestamp': '2026-01-08T11:54:50.417Z'}


    env_id = 'env_7c87bb26138a427eb93cab27d0f5429f'
    # env_data must be the env config (with "modules") - same shape as orchestrator passes
    env_data = payload["data"]["config"][env_id]
    g = GUtils()
    qfu = QFUtils(g)
    g = Guard(
        qfu=qfu,
        g=g,
        user_id="72b74d5214564004a3a86f441a4a112f",
    )
    g.main(env_id=env_id, env_data=env_data)




async def run_start_sim_via_ws():
    user_id = payload["auth"]["user_id"]
    uri = f"ws://127.0.0.1:8001/run/?user_id={user_id}"
    print("Running START_SIM test via WebSocket...")
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            await websocket.send(json.dumps(payload))
            print("Sent START_SIM payload")
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    msg = json.loads(response)
                    print(f"Received: type={msg.get('type')}, status={msg.get('status', {})}")
                    if msg.get("type") == "START_SIM" and msg.get("status", {}).get("state") in ("success", "error"):
                        break
                except asyncio.TimeoutError:
                    print("No further messages (timeout)")
                    break
    except Exception as e:
        print(f"Failed: {e}")


"""
CE VM: create cfg -> create & exec
cfg = self.deployment_handler.get_prod_vm_cfg(
    env_id,
    latest_image,
    container_env,
    testing=self.testing,
)
print(f"  -> VM-Payload bereit (Image: {latest_image})")
# 5. Deployment Call (deploy_vms expects {key: cfg} dict)
vm_payload = {env_id: cfg}
self.deploy_vms(vm_payload)
"""

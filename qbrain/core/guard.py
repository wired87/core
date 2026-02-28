import base64
import os
import pprint

from itertools import product

import numpy as np
from _admin.bob_builder.artifact_registry.artifact_admin import ArtifactAdmin
import networkx as nx
import asyncio
import json
import websockets
from qbrain.code_manipulation.graph_creator import StructInspector
from qbrain.graph.local_graph_utils import GUtils
from qbrain.qf_utils.qf_utils import QFUtils
from qbrain.a_b_c.bq_agent._bq_core.bq_handler import BQCore

from qbrain.core.qbrain_manager import get_qbrain_table_manager, QBrainTableManager

from qbrain.core.module_manager.mcreator import ModuleCreator
from qbrain.utils.math.operator_handler import EqExtractor
from qbrain.qf_utils.all_subs import ALL_SUBS
from qbrain.utils._np.expand_array import expand_structure
from qbrain.utils.get_shape import extract_complex, get_shape
from qbrain.utils.xtract_trailing_numbers import extract_trailing_numbers
#from vertex_trainer.manager import VertexTrainerManager
from qbrain.workflows.deploy_sim import DeploymentHandler
from qbrain.utils.run_subprocess import pop_cmd


from qbrain.core.fields_manager.fields_lib import FieldsManager
from qbrain.core.method_manager.method_lib import MethodManager

from qbrain.core.injection_manager import InjectionManager
from qbrain.core.module_manager.ws_modules_manager import ModuleWsManager
from qbrain.core.param_manager.params_lib import ParamsManager

from qbrain.core.env_manager.env_lib import EnvManager


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
        field_manager,
        method_manager,
        injection_manager,
        env_manager,
        module_db_manager,
        params_manager,
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
        self.params_manager = params_manager
        self.deployment_handler = DeploymentHandler(
            user_id
        )

        self.qb = get_qbrain_table_manager()
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

        self.eq_extractor = EqExtractor()
        #self.operator_handler = OperatorHandler()

        self.prev_state = None
        self.model_params = None
        self.fnished_modules = False

        self.code_extractor = StructInspector(
            G=self.g.G,
        )

        self.fields = []
        print("Guard Initialized!")

    # ---- NODE / EDGE HELPERS ----
    def _node(self, nid: str, ntype: str, **kwargs):
        """Add a node. ntype: ENV, MODULE, FIELD, INJECTION, METHOD, PARAM."""
        self.g.add_node({"id": nid, "type": ntype.upper(), **kwargs})

    def _edge(self, src: str, trt: str, rel: str, src_layer: str, trgt_layer: str):
        """Add an edge with standard attrs."""
        self.g.add_edge(src, trt, attrs={
            "rel": rel, "src_layer": src_layer, "trgt_layer": trgt_layer
        })

    def create_fields(self, field_ids):
        """Fetch fields from BQ and add FIELD nodes."""
        if not field_ids:
            return
        res = self.qb.row_from_id(list(field_ids), table=self.field_manager.FIELDS_TABLE)
        for i, item in enumerate(res):
            fid = item["id"].upper()
            if item.get("interactant_fields") and isinstance(item["interactant_fields"], str):
                try:
                    item["interactant_fields"] = json.loads(item["interactant_fields"])
                except json.JSONDecodeError:
                    pass
            extra = {k: v for k, v in item.items() if k not in ("id", "id", "type")}
            extra["field_index"] = i
            self._node(fid, "FIELD", **extra)

    def _collect_injections(self, env_data: dict) -> dict:
        """Collect injections from env_data. Supports top-level or nested under modules->fields."""
        inj = env_data.get("injections", {})
        if inj:
            return inj
        out = {}
        for mod_data in (env_data.get("modules") or {}).values():
            for field_id, field_cfg in (mod_data.get("fields") or {}).items():
                i = field_cfg.get("injections", {})
                if i:
                    out[field_id.upper()] = i
        return out

    def create_nodes(self, env_id, env_data):
        """Reset graph and create ENV, MODULE, FIELD, INJECTION nodes."""
        self.g.G = nx.Graph()
        module_ids = set()
        field_ids = set()

        self._node(env_id, "ENV")

        module_ids_raw = env_data.get("modules", [])
        if isinstance(module_ids_raw, dict):
            module_ids_raw = list(module_ids_raw.keys())
        module_ids_raw = list(dict.fromkeys(str(m).upper() for m in module_ids_raw))
        res = self.qb.row_from_id(module_ids_raw, table=self.module_db_manager.MODULES_TABLE) if module_ids_raw else []
        fetched = {item["id"].upper(): item for item in res}

        for i, (mod_id, mod) in enumerate(fetched.items()):
            module_ids.add(mod_id)
            self._node(mod_id, "MODULE", module_index=i, **{k: v for k, v in mod.items() if k not in ["id", "module_index"]})
            field_list = mod.get("fields") or []

            if isinstance(field_list, str):
                try:
                    field_list = json.loads(field_list) or []
                except json.JSONDecodeError:
                    field_list = []
            if isinstance(field_list, dict):
                field_list = list(field_list.keys())
            field_ids.update(str(f).upper() for f in field_list)

        self.create_fields(field_ids)
        inj_ids = self._add_injections(self._collect_injections(env_data))
        return module_ids, field_ids, inj_ids

    def _add_injections(self, injections: dict):
        """Add INJECTION nodes and FIELD->INJECTION edges."""
        inj_ids = set()
        for field_id, data in injections.items():
            for pos, inj_id in data.items():
                inj_ids.add(inj_id)
                nid = f"{pos}__{inj_id}"
                self._node(nid, "INJECTION")
                self._edge(field_id.upper(), nid, "has_injection", "FIELD", "INJECTION")
        return inj_ids

    def create_edges(self, env_id, env_data):
        """Add ENV->MODULE, MODULE->FIELD edges and param links."""
        modules = env_data.get("modules", {})

        if isinstance(modules, list):
            for mod_id in modules:
                mod_id = str(mod_id).upper()
                if not self.g.G.has_node(mod_id):
                    continue

                self._edge(env_id, mod_id, "has_module", "ENV", "MODULE")

                try:
                    mod_node = self.g.get_node(mod_id)
                    field_list = mod_node.get("fields") or []
                    if isinstance(field_list, str):
                        field_list = json.loads(field_list) if field_list else []
                    if isinstance(field_list, dict):
                        field_list = list(field_list.keys())
                    for fid in field_list:
                        fid = str(fid).upper()
                        self._edge(mod_id, fid, "has_field", "MODULE", "FIELD")
                        try:
                            fn = self.g.get_node(fid)
                            self.qfu.add_params_link_fields(
                                fn.get("keys", []),
                                fn.get("values", []),
                                fid,
                                mod_id
                            )
                        except Exception as e:
                            print(f"Error linking params for {fid}: {e}")
                except Exception as e:
                    print(f"Error creating edges for module {mod_id}: {e}")
            return

        for mod_id, mod_data in modules.items():
            mod_id = mod_id.upper()
            self._edge(env_id, mod_id, "has_module", "ENV", "MODULE")
            for field_id, field_data in mod_data.get("fields", {}).items():
                field_id = field_id.upper()
                self._edge(mod_id, field_id, "has_field", "MODULE", "FIELD")
                try:
                    fn = self.g.get_node(field_id)
                    self.qfu.add_params_link_fields(
                        fn.get("keys", []), fn.get("values", []), field_id, mod_id
                    )
                except Exception as e:
                    print(f"Error linking params for {field_id}: {e}")


    def check_create_ghost_mod(self, env_id):
        """Add GHOST_MODULE node and ENV->GHOST_MODULE edge."""
        modules = self.g.get_nodes(filter_key="type", filter_value="MODULE", just_id=True)
        self._node("GHOST_MODULE", "MODULE", module_index=len(modules), status="ghost")
        self._edge(env_id, "GHOST_MODULE", "has_module", "ENV", "MODULE")


    def handle_env(self, env_id):
        """Enrich ENV node with params; add ENV as FIELD under GHOST_MODULE for param collection."""
        try:
            keys = list(self.qfu.create_env().keys())
            env_constants = self.qb.row_from_id(keys, table="params")
            values = [v["value"] for v in env_constants]
            keys = [v["id"] for v in env_constants]
            axis_def = [v["axis_def"] for v in env_constants]
        except Exception as e:
            print(f"[FIX] handle_env params fetch failed: {e}, using defaults")
            keys, values, axis_def = ["None"], [None], [None]
        values.append(None)
        keys.append("None")
        axis_def.append(None)

        res_list = self.qb.row_from_id([env_id], table=self.env_manager.TABLE_ID)
        res = res_list[0] if res_list else {}
        ghost_fields = self.g.get_neighbor_list_rel(node="GHOST_MODULE", trgt_rel="has_field")
        self.g.add_node(dict(
            id=env_id, type="FIELD", sub_type="ENV", keys=keys, values=values, axis_def=axis_def,
            field_index=len(ghost_fields), **res
        ))
        self._edge("GHOST_MODULE", env_id, "has_field", "MODULE", "FIELD")

    def handle_field_interactants(self, field_ids: list[str], env_id: str):
        """Add missing interactant FIELD nodes and FIELD->FIELD (has_finteractant) edges."""
        for fid in field_ids:
            fattrs = self.g.get_node(fid)
            interactant_fields = fattrs.get("interactant_fields") or []
            if isinstance(interactant_fields, str):
                interactant_fields = json.loads(interactant_fields) if interactant_fields else []

            missing = [fi for fi in interactant_fields if not self.g.G.has_node(fi)]
            if missing:
                res = self.qb.row_from_id(missing, table=self.env_manager.TABLE_ID)
                for item in res:
                    mfid = item["id"]
                    vals = item.get("values")
                    keys = item.get("keys")
                    if isinstance(vals, str):
                        vals = json.loads(vals) if vals else []
                    if isinstance(keys, str):
                        keys = json.loads(keys) if keys else []
                    extra = {k: v for k, v in item.items() if k not in ("values", "keys", "id")}
                    self._node(mfid, "FIELD", keys=keys, values=vals, **extra)
                    mod_id = item.get("module_id")
                    if mod_id and self.g.G.has_node(mod_id):
                        self._edge(mod_id, mfid, "has_field", "MODULE", "FIELD")
                    elif mod_id:
                        self.check_create_ghost_mod(env_id)

            for mfid in interactant_fields:
                self._edge(fid, mfid, "has_finteractant", "FIELD", "FIELD")

    def main(self, env_id, env_data, cfg_path=None, grid_streamer=None, grid_animation_recorder=None):
        print("main...")
        #pprint.pp(env_data)
        self.data_handler(env_id, env_data)
        components = self.converter(env_id)
        print("components", components)
        #self.handle_deployment(env_id, components)

        if grid_streamer is not None and components:
            self._push_grid_frame(components, grid_streamer, step=0)
        if grid_animation_recorder is not None and components:
            self._save_animation_frame(components, grid_animation_recorder, step=0)

        if os.getenv("LOCAL_DB", "True") == "True":
            project_root = os.path.dirname(os.path.dirname(__file__))
            cfg_file = cfg_path or os.getenv("GRID_CFG_PATH", "test_out.json")
            if not os.path.isabs(cfg_file):
                cfg_file = os.path.join(project_root, cfg_file)
            with open(cfg_file, "w") as f:
                f.write(json.dumps(components, indent=4))
            print(f"[LOCAL_DB] cfg written to {cfg_file}")

            import sys
            grid_cmd = os.getenv("GRID_CMD", "{python} -m jax_test.grid --cfg {cfg_path}")
            cmd = (
                grid_cmd.replace("{cfg_path}", cfg_file)
                .replace("{cfg}", cfg_file)
                .replace("{python}", sys.executable)
            )
            print(f"[LOCAL_DB] running grid workflow: {cmd}")
            try:
                pop_cmd(cmd, cwd=project_root)
            except Exception as e:
                print(f"[LOCAL_DB] grid workflow error: {e}")
                raise

            # Save model path to envs table
            model_path = os.getenv("GRID_MODEL_OUT", os.path.join(project_root, "model_out.json"))
            if os.path.isfile(model_path):
                self._save_model_path_to_envs(env_id, model_path)
            else:
                print(f"[guard] model file not found, skipping envs update: {model_path}")

        if grid_animation_recorder is not None:
            grid_animation_recorder.finish()
        print("main... done")

    def handle_deployment(self, env_id, components):
        print(f"\n[START] handle_deployment für Env: {env_id}")
        try:
            if not self._is_components_valid_for_grid(components):
                print("  [WARN] Aborting deployment: components have empty/invalid structures for grid-root")
                return
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
            from qbrain.core.managers_context import get_user_manager
            get_user_manager().qb.set_item(
                "envs",
                {"status": "IN_PROGRESS"},
                keys={"id": env_id, "user_id": self.user_id}
            )
            print("  -> Status in DB auf 'IN_PROGRESS' gesetzt")

            print(f"  -> Modus: TESTING (Speichere lokal)")
            import os
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_out.json")
            with open(path, "w") as f:
                f.write(json.dumps(components, indent=4))
            print(f"  -> Datei erfolgreich geschrieben unter: {path}")

        except Exception as e:
            print(f"  [!!!] ERROR in handle_deployment: {e}")
            import traceback
            traceback.print_exc()
            try:
                from qbrain.core.managers_context import get_user_manager
                get_user_manager().qb.set_item(
                    "envs",
                    {"status": "FAILED"},
                    keys={"id": env_id, "user_id": self.user_id}
                )
            except Exception as e:
                pass


    def create_init_param_nodes(self, fid, param_ids):
        """Add PARAM nodes and FIELD->PARAM edges."""
        if not param_ids:
            return
        res = self.qb.row_from_id(param_ids, table=self.params_manager.PARAMS_TABLE)
        for item in res:
            pid = item["id"]
            self._node(pid, "PARAM", param_type=item.get("param_type"), value=item.get("value"))
            self._edge(fid, pid, "has_param", "FIELD", "PARAM")



    def _validate_env_data(self, env_id: str, env_data: dict) -> None:
        """Ensure no cfg entries in env_data are empty. Raises ValueError if invalid."""
        if not env_data:
            raise ValueError(f"[guard] env_data for {env_id} is empty")
        modules = env_data.get("modules")
        if modules is None:
            raise ValueError(f"[guard] env_data.modules is missing for {env_id}")
        if isinstance(modules, list) and len(modules) == 0:
            raise ValueError(f"[guard] env_data.modules is empty for {env_id}")
        if isinstance(modules, dict) and len(modules) == 0:
            raise ValueError(f"[guard] env_data.modules is empty for {env_id}")
        # injections may be empty {} when no injections; only modules must be non-empty

    def _push_grid_frame(self, components: dict, grid_streamer, step: int = 0) -> None:
        """Decode DB from components and push to grid streamer (non-blocking)."""
        db_b64 = components.get("DB")
        if not db_b64:
            return
        try:
            raw = base64.b64decode(db_b64)
            arr = np.frombuffer(raw, dtype=np.complex64)
            data = np.abs(arr).astype(np.float32) if np.iscomplexobj(arr) else arr.astype(np.float32)
            grid_streamer.put_frame(step, data)
        except Exception as e:
            print(f"[guard] _push_grid_frame: {e}")

    def _save_animation_frame(self, components: dict, recorder, step: int = 0) -> None:
        """Decode DB and save plot frame for animation recorder."""
        db_b64 = components.get("DB")
        if not db_b64:
            return
        try:
            raw = base64.b64decode(db_b64)
            arr = np.frombuffer(raw, dtype=np.complex64)
            data = np.abs(arr).astype(np.float32) if np.iscomplexobj(arr) else arr.astype(np.float32)
            recorder.cfg = components
            recorder.save_frame(step, data)
        except Exception as e:
            print(f"[guard] _save_animation_frame: {e}")

    def _save_model_path_to_envs(self, env_id: str, model_path: str) -> None:
        """Update envs table row with model_path (and npz_path)."""
        try:
            if hasattr(self.env_manager.qb, "insert_col"):
                try:
                    self.env_manager.qb.insert_col(
                        self.env_manager.TABLE_ID,
                        "model_path",
                        "STRING",
                    )
                except Exception:
                    pass
            model_path_abs = os.path.abspath(model_path)
            npz_path = model_path_abs.replace(".json", "_data.npz")
            updates = {"model_path": model_path_abs}
            if os.path.isfile(npz_path):
                try:
                    self.env_manager.qb.insert_col(
                        self.env_manager.TABLE_ID,
                        "model_data_path",
                        "STRING",
                    )
                except Exception:
                    pass
                updates["model_data_path"] = npz_path
            self.env_manager.qb.set_item(
                self.env_manager.TABLE_ID,
                updates,
                keys={"id": env_id, "user_id": self.user_id},
            )
            print(f"[guard] saved model_path to envs: {model_path_abs}")
        except Exception as e:
            print(f"[guard] _save_model_path_to_envs error: {e}")

    def data_handler(self, env_id, env_data):
        """
        1. Parse payload to get IDs.
        2. Batch fetch data.
        3. Convert to pattern (Graph).
        4. Compile pattern.
        """
        print("sim_start_process...")
        self._validate_env_data(env_id, env_data)

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


        # hadler
        self.handle_env(env_id)
        self.handle_methods(module_ids=list(module_ids))
        self.handle_field_interactants(
            list(field_ids),
            env_id
        )


        # ETEND FIELDS PARAMS WITH RETURN KEYS
        self.extend_fields_keys()

        self.get_inj_data(inj_ids)

        print("data_hanadler... done")


    def get_inj_data(self, inj_ids):
        # Enrich INJECTION nodes with BQ data
        if inj_ids:
            print("Fetching injections from BQ...")

            res = self.qb.row_from_id(
                list(inj_ids),
                table=self.injection_manager.table
            )
        
            fetched_injections = {item["id"]: item for item in res}

            for inid, inj_data in fetched_injections.items():
                for nid, attrs in self.g.G.nodes(data=True):
                    if attrs.get("type") == "INJECTION" and inid in nid and inj_data:
                        extra = {k: v for k, v in inj_data.items() if k not in ("id", "type")}
                        self._node(nid, "INJECTION", **extra)
            print(f"Fetched injections: {len(fetched_injections)}")



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

        # create emthod strucutre to fill with
        #self.operator_handler.method_schema = self.get_empty_method_structure()


        # create iterator last
        iterators = self.set_iterator_from_humans()

        db_to_method_struct:dict = self.set_edge_db_to_method(modules, env_id)

        method_struct:dict = self.method_layer(
            modules,
            #db_to_method_struct["VARIATION_INDICES"],
        )

        method_to_db = self.set_edge_method_to_db()

        injection_patterns, flatten_e_map = self.handle_energy_components(env_node)

        components = {
            **self.DB,
            **iterators,
            **method_struct,
            **injection_patterns,
            **db_to_method_struct,
            "METHOD_TO_DB": method_to_db if method_to_db is not None else [],
            "E_KEY_MAP_PER_FIELD": flatten_e_map if flatten_e_map is not None else [],
            # include serialized OPERATORS
            #"OPERATOR": serialize_ops()
        }

        components = self._sanitize_components(components)
        local_db = os.getenv("LOCAL_DB", "True") == "True"
        self._validate_components_no_empty(components, strict=not local_db)
        print("Main... done")
        return components

    def _sanitize_components(self, components: dict) -> dict:
        """Ensure no None or invalid empty structures that would break grid-root."""
        out = dict(components)
        list_keys = [
            "DB", "AXIS", "DB_SHAPE", "AMOUNT_PARAMS_PER_FIELD", "DB_PARAM_CONTROLLER",
            "DB_KEYS", "FIELD_KEYS", "MODULES", "FIELDS",
            "DB_TO_METHOD_EDGES", "METHOD_TO_DB", "DB_CTL_VARIATION_LEN_PER_EQUATION",
            "DB_CTL_VARIATION_LEN_PER_FIELD", "LEN_FEATURES_PER_EQ",
            "METHOD_PARAM_LEN_CTLR", "METHODS_PER_MOD_LEN_CTLR", "NEIGHBOR_CTLR",
            "INJECTOR_TIME", "INJECTOR_INDICES", "INJECTOR_VALUES",
            "E_KEY_MAP_PER_FIELD",
        ]
        for k in list_keys:
            if k in out and out[k] is None:
                out[k] = []
        if "METHOD_TO_DB" in out and not isinstance(out["METHOD_TO_DB"], list):
            out["METHOD_TO_DB"] = []
        if "DB_TO_METHOD_EDGES" in out and not isinstance(out["DB_TO_METHOD_EDGES"], list):
            out["DB_TO_METHOD_EDGES"] = []
        return out

    _OPTIONAL_EMPTY_KEYS = frozenset({
        "INJECTOR_TIME", "INJECTOR_INDICES", "INJECTOR_VALUES",
        "E_KEY_MAP_PER_FIELD",
    })

    def _validate_components_no_empty(self, components: dict, strict: bool = True) -> None:
        """Ensure no cfg entry in components is empty. Raises ValueError with details when strict.
        Keys in _OPTIONAL_EMPTY_KEYS may be empty (e.g. when no injections).
        When strict=False (e.g. LOCAL_DB), logs warning instead of raising."""
        if not components:
            if strict:
                raise ValueError("[guard] components dict is empty")
            print("[guard] WARN: components dict is empty")
            return
        empty_keys = []
        for k, v in components.items():
            if k in self._OPTIONAL_EMPTY_KEYS:
                continue
            if v is None:
                empty_keys.append(f"{k}=None")
            elif isinstance(v, (list, dict)) and len(v) == 0:
                empty_keys.append(f"{k}=[]/{{}}")
            elif isinstance(v, str) and v.strip() == "":
                empty_keys.append(f"{k}=''")
        if empty_keys:
            msg = f"[guard] cfg entries must not be empty. Empty: {', '.join(empty_keys)}"
            if strict:
                raise ValueError(msg)
            print(f"[guard] WARN: {msg}")

    def _is_components_valid_for_grid(self, components: dict) -> bool:
        """Check if components have minimal non-empty structure for grid-root."""
        if not components:
            return False
        if components.get("DB") is None:
            return False
        if "AXIS" not in components or "FIELDS" not in components:
            return False
        modules = components.get("MODULES", [])
        fields = components.get("FIELDS", [])
        if not modules or not fields:
            return False
        return True


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


    def handle_energy_components(
            self,
            env_attrs,
            trgt_keys=["energy", "j_nu", "vev"], # injectable parameters
    ):
        # exact same format
        print("handle_energy_components...")

        INJECTOR = {
            "INJECTOR_TIME":[],
            "INJECTOR_INDICES":[],
            "INJECTOR_VALUES":[],
        }

        flatten_e_map = []

        try:
            amount_nodes = env_attrs.get("amount_of_nodes", 1)
            dims = env_attrs.get("dims", 3)
            schema_positions = self.get_positions(amount_nodes, dims)

            modules = self.g.get_nodes(
                filter_key="type",
                filter_value="MODULE"
            )

            E_KEY_MAP_PER_FIELD = [[] for _ in range(len(modules))]

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

                            E_KEY_MAP_PER_FIELD[midx].append(
                                field_rel_param_trgt_index
                            )

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

                                # check str
                                if isinstance(inj_attrs["data"], str):
                                    inj_attrs["data"] = json.loads(inj_attrs["data"])

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

            # flatten E_KEY_MAP_PER_FIELD
            for mod in E_KEY_MAP_PER_FIELD:
                flatten_e_map.extend(mod)
            print(f"handle_energy_components... done")
        except Exception as e:
            print("Err handle_energy_components", e)
        return INJECTOR, flatten_e_map




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
                # [FIX] ensure list has slot for m_idx
                while len(db) <= m_idx:
                    db.append([])
                    axis.append([])
                    shapes.append([])
                    item_len_collection.append([])
                    param_len_collection.append([])
                    field_ids.append([])
                    db_keys.append([])

                print(f"[create_db] module={mid}, idx={m_idx}")

                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True
                )

                for fid, fattrs in fields.items():
                    fidx = fattrs["field_index"]
                    # [FIX] ensure field slot exists
                    while len(db[m_idx]) <= fidx:
                        db[m_idx].append([])
                        axis[m_idx].append([])
                        shapes[m_idx].append([])
                        item_len_collection[m_idx].append([])
                        param_len_collection[m_idx].append(0)
                        field_ids[m_idx].append(None)
                        db_keys[m_idx].append([])

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


    def get_execution_order(self, method_definitions: list) -> list:
        """
        Determines the correct execution order of methods based on return_key dependencies.

        Expects method dicts with schema: id, return_key, params (str or list).
        params can be JSON string; parse to list for dependency check.

        Returns:
            List of method dicts in dependency order.
        """
        import json

        def _parse_params(m):
            p = m.get("params")
            if p is None:
                return []
            if isinstance(p, list):
                return p
            if isinstance(p, str):
                try:
                    return json.loads(p) if p.strip() else []
                except (json.JSONDecodeError, AttributeError):
                    return []
            return []

        # Identify all keys produced by any method
        internal_returns = {m["return_key"] for m in method_definitions if m.get("return_key")}

        scheduled_order = []
        produced_keys = set()
        remaining = list(method_definitions)

        while remaining:
            ready = []
            for m in remaining:
                params = _parse_params(m)
                internal_deps = set(params).intersection(internal_returns)
                if internal_deps.issubset(produced_keys):
                    ready.append(m)

            if not ready:
                break

            for m in ready:
                scheduled_order.append(m)
                if m.get("return_key"):
                    produced_keys.add(m["return_key"])
                remaining.remove(m)

        # Append any unscheduled (e.g. cyclic or orphan) methods
        scheduled_order.extend(remaining)
        return scheduled_order


    def handle_methods(self, module_ids:list[str]):
        print("handle_methods...")
        import json
        method_idx = 0
        for mid in module_ids:
            mod_node = self.g.get_node(id=mid)
            mmethods = mod_node["methods"]

            if isinstance(mmethods, str):
                mmethods = json.loads(mmethods)

            methods = self.qb.row_from_id(
                id=mmethods,
                table=self.method_manager.METHODS_TABLE,
            )

            methods = {
                item["id"]: item
                for item in methods
            }

            # Sort by execution order (return_key dependencies)
            method_list = list(methods.values())
            ordered_methods = self.get_execution_order(method_list)

            for method_data in ordered_methods:
                method_id = method_data["id"]
                extra = {k: v for k, v in method_data.items() if k not in ("id", "type")}
                self._node(method_id, "METHOD", method_idx=method_idx, **extra)
                self._edge(mid, method_id, "has_method", "MODULE", "METHOD")
                method_idx += 1
        print("handle_methods... done")


    def create_method_param_nodes(self, modules):
        """
        todo before: fetch and add param nodes include type
        Create Param nodes form fiedls
        Goal:
        """
        for mid, module in modules:
            # ghost does not have equation
            if "GHOST" in mid.upper(): continue
            # print("method_layer... working", mid)

            fields = self.g.get_neighbor_list_rel(
                trgt_rel="has_field",
                node=mid,
                as_dict=True,
            )
            for fid, fattrs in fields.items():
                # PARAMS from METHOD
                values = fattrs.get("fields", [])
                keys = fattrs.get("keys", [])

                for param, value in zip(keys, values):
                    # type already exists
                    self.g.update_node({
                        "id": param,
                        "type": "PARAM",
                        "value": value,
                    })

                    self._edge(fid, param, "has_param", "FIELD", "PARAM")

    def is_differnetial_equation(self, params):
        normalized = [p.replace("_", "") for p in params]
        has_duplicates = any(normalized.count(np) == 2 for np in set(normalized) if np)

        # Zusammenführung
        if has_duplicates and self.has_special_params(params):
            return True
        return False

    def has_special_params(self, params):
        return any(p.endswith("__") for p in params) or any(p.startswith("_prev") or p.startswith("prev_") for p in params)


    def is_interaction_eq(self, params, modules_params:list[str], modules_return_map:list[str]):
        normalized = [p.replace("_", "") for p in params]
        has_duplicates = any(normalized.count(np) == 2 for np in set(normalized) if np)
        prefixed_dublet = any(pid.endswith("_") for pid in params) and any(pid.startswith("_") for pid in params)
        params_of_different_fields = any(p not in modules_params for p in params) or any(p not in modules_return_map for p in params)
        if has_duplicates or prefixed_dublet or params_of_different_fields:
            return True
        return False

    def classify_equations_for_module(
        self,
        methods: dict,
        fields: dict,
    ) -> dict:
        """
        Classify equations for a specific module into a dict with keys:
        - differential: method includes same param min 2 times (param.replace("_","") for param in params)
        - interaction: method params originate from min 2 fields of different types
        - core: method requires params of just single field type, or uses return_key of other methods in the module
        """
        classification = {"differential": [], "interaction": [], "core": []}
        module_return_keys = [eqattrs.get("return_key") for eqattrs in methods.values() if eqattrs.get("return_key")]
        modules_params = [fattrs.get("keys") for fattrs in fields.values()]

        for eqid, eqattrs in methods.items():
            params = eqattrs.get("params")
            _code = eqattrs.get("code")
            if self.is_differnetial_equation(params):
                classification["differential"].append(_code)

            elif self.is_interaction_eq(params, modules_params, module_return_keys):
                classification["interaction"].append(_code)
            else:
                classification["core"].append(_code)
        return classification

    def method_layer(self, modules):
        # For each method: use params and neighbor_vals to collect the params index for each item of
        # neighbor_vals within a list (if neighbor_vals else None), and append it to a NEIGHBOR_CTLR
        # struct under the same index as the specific method (and overlying module_idx).
        print("method_layer... ")
        mod_len_exclude_ghost = len(modules) -1
        #print("method_layer mod_len_exclude_ghost", mod_len_exclude_ghost)

        method_struct = {
            "METHOD_PARAM_LEN_CTLR": [[] for _ in range(mod_len_exclude_ghost)],
            "METHODS": [[] for _ in range(mod_len_exclude_ghost)],
            "METHODS_PER_MOD_LEN_CTLR": [0 for _ in range(mod_len_exclude_ghost)],
            "NEIGHBOR_CTLR": [[] for _ in range(mod_len_exclude_ghost)],
            "EQ_CLASSIFICATION": [[] for _ in range(mod_len_exclude_ghost)],
        }

        mnames = [[] for _ in range(mod_len_exclude_ghost)]

        try:
            #print("method_layer compilation...")
            for mid, module in modules:
                # ghost does not have equation
                if "GHOST" in mid.upper(): continue

                midx:int = module.get("module_index")

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

                fields = self.g.get_neighbor_list_rel(
                    node=mid,
                    trgt_rel="has_field",
                    as_dict=True,
                )

                # Classification struct per module
                method_struct["METHODS"][midx] = self.classify_equations_for_module(
                    methods,
                    fields,
                )

                # len methods per module
                method_struct["METHODS_PER_MOD_LEN_CTLR"][midx] = mlen
                mnames[midx].extend(mids)

            flatten_ctlr = []
            for sublist in method_struct["METHOD_PARAM_LEN_CTLR"]:
                flatten_ctlr.extend(sublist)
            method_struct["METHOD_PARAM_LEN_CTLR"] = flatten_ctlr

            flatten_neighbor_ctlr = []
            for sublist in method_struct["NEIGHBOR_CTLR"]:
                flatten_neighbor_ctlr.append(sublist)
            method_struct["NEIGHBOR_CTLR"] = flatten_neighbor_ctlr

            print("METHODS raw", method_struct["METHODS"])
            flatten_eq_classification_ctlr = []
            for struct in method_struct["METHODS"]:
                flatten_eq_classification_ctlr.append(
                    [v.values() for v in struct]
                )
            method_struct["METHODS"] = flatten_eq_classification_ctlr
            print("METHODS processed", method_struct["METHODS"])

        except Exception as e:
            print("Err method_layer", e)
        print(f"method_layer... done")
        return method_struct



    def set_eq_operator_ctlr(self, modules):
        # print("method_layer compilation...")
        for mid, module in modules:
            # ghost does not have equation
            if "GHOST" in mid.upper(): continue
            # print("method_layer... working", mid)

            midx: int = module.get("module_index")

            methods = self.g.get_neighbor_list_rel(
                trgt_rel="has_method",
                node=mid,
                as_dict=True,
            )

            mids = list(methods.keys())
            mlen: int = len(mids)

            if not mlen:
                print("method_layer... len methods", mlen)
                continue

            # Iterate Equations
            for eqid, eqattrs in methods.items():
                params = eqattrs.get("params")
                equation = eqattrs.get("equation")
                print("eq extractted", equation)

                # set operator map eq based
                self.operator_handler.process_code(
                    code=equation,
                    params=params,
                    midc=midx
                )
        return self.operator_handler.operator_pathway_ctlr










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
            # Return valid empty structure to avoid None breaking grid
            return []



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
            #"VARIATION_KEYS": [[] for _ in range(mlen)],
            #"VARIATION_INDICES": [[] for _ in range(mlen)]
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

                        # LOOP EQ-PARAMS

                        # todo if 3 (or more) identical keys include dict field, bool - map
                        worked_params = {}
                        for pidx, pid in enumerate(params):
                            pid_orig = pid  # keep original for worked_params lookup
                            if pid not in worked_params:
                                # space collect fields
                                worked_params[pid] = []

                            collected = False

                            param_collector = []
                            param_origin_key_collector = []
                            #print("work pid", pid)

                            # Field's own param
                            is_prefixed = pid.endswith("_")
                            is_self_prefixed = pid.startswith("_")
                            is_prev_pre = pid.startswith("prev_")
                            is_prev_after = pid.endswith("_prev")

                            # is_double_after_marked = pid.endswih("__")

                            EXCLUDED_ORIGINS = ["neighbor", "interactant"]

                            if (
                                pid in keys and
                                    (not is_prefixed or is_self_prefixed) and
                                    params_origin[pidx] not in EXCLUDED_ORIGINS and
                                    fid not in worked_params[pid_orig] or not is_self_prefixed or (is_prev_pre or is_prev_after)
                            ):
                                print(f"{pid} in {fid}")

                                # Add field to param collection
                                worked_params[pid_orig].append(fid)

                                time_dim = None

                                # RM start "_"
                                if is_prev_pre:
                                    #print("prev etected:", pid)
                                    # _prev must be in keys!!!
                                    pid = pid.replace("prev_","").strip()
                                    time_dim = 1
                                elif is_prev_after:
                                    pid = pid.replace("_prev","").strip()
                                    time_dim = 1

                                elif is_self_prefixed:
                                    pid = pid[:-1]

                                # directly sort in param arrays
                                if time_dim is None:
                                    time_dim = 0


                                # todo problem: params of method apply to field keys
                                # A: SMManager: params liked tomodule not in field (for each) append to it -> how infer type? scale to nDim struct?
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
                                    if pid in ikeys and finid not in worked_params[pid_orig]:

                                        # Add field to param collection
                                        worked_params[pid_orig].append(fid)

                                        fmod = self.g.get_node(id=fiattrs.get("module_id"))

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

                                    if pid in gikeys and gfid not in worked_params[pid_orig]:

                                        # Add field to param collection
                                        worked_params[pid_orig].append(fid)

                                        gfield_index = gfattrs["field_index"]

                                        pindex = gikeys.index(pid)
                                        #print("interactant pindex", pindex)
                                        #print(f"{pid} found in gfid ({gfield_index}) (pindex {pindex})", gfield_index)

                                        gmod = self.g.get_node(id="GHOST_MODULE")

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
                                    gmod = self.g.get_node(id="GHOST_MODULE")
                                    env_field = self.g.get_node(id=env_id)
                                    pindex = env_field["keys"].index("None")
                                    param_collector.append(
                                        self.get_db_index(
                                            gmod["module_index"],
                                            env_field["field_index"],
                                            pindex,
                                        )
                                    )

                            field_eq_param_struct.append(param_collector)

                        # get scaled soa [[a,a], [b,b]
                        expand_field_eq_variation_struct = expand_structure(
                            struct=field_eq_param_struct
                        )

                        param_struct = {}
                        for key, value in zip(params, expand_field_eq_variation_struct):
                            param_struct[key] = value

                        #db_to_method["VARIATION_INDICES"][m_idx].append(param_struct)

                        # todo: untertiele method nach modul index.
                        #  mach heir das gleiche und flatte im Anschlusse

                        # extend variation single eq
                        for item in expand_field_eq_variation_struct:
                            db_to_method["DB_TO_METHOD_EDGES"][m_idx].extend(item)
                            #print(f"module {mid} ({m_idx}) expand_field_eq_variation_struct item {eqid}", item)
                            # todo calc just / len(method_param) to sort them

                        field_variations_eq = len(expand_field_eq_variation_struct)
                        EQ_AMOUNT_VRIATIONS += field_variations_eq

                        # add len field var to emthod struct
                        db_to_method[
                            "LEN_FEATURES_PER_EQ"
                        ][eq_idx].append(field_variations_eq)

                        db_to_method[
                            "DB_CTL_VARIATION_LEN_PER_FIELD"
                        ][m_idx].append(len(expand_field_eq_variation_struct))

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
                #m_idx = module.get("module_index")
                # A. FeDB_PARAM_CONTROLLERlder des Moduls sammeln
                fields = self.g.get_neighbor_list_rel(
                    node=mid, trgt_rel="has_field", as_dict=True
                )
                len_fields = len(fields)
                iterator["FIELDS"].append(len_fields)

                # B. Methoden (Gleichungen) des Moduls sammeln
                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method", node=mid, as_dict=True
                )
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
                                        pmod = self.g.get_node(id=fiattrs["module_id"])
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
                                        pmod = self.g.get_node(id="GHOST_MODULE")

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
        except Exception as e:
            print(f"Error in set_iterator_from_humans: {e}")
            raise e
        return iterator


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

    def sync_field_keys_from_methods(self, dims: int = None):
        # Implement a method within Guard that collects all keys for each field summed in each module,
        # -> collects all params incl return_key of each method per module
        # -> checks which keys in fields does not exist
        # -> adds the key to the keys list of each modules specific field neighbor and applies a default value
        # (list[0 for _ in range(dims)], axis_def of 0 to the field)
        """
        Sync field keys from method params and return_key. Adds missing keys to each module's field
        neighbors with default value [0]*dims and axis_def=0.
        """
        print("sync_field_keys_from_methods...")
        try:
            # Resolve dims from ENV node or default
            if dims is None:
                env_nodes = self.g.get_nodes(filter_key="type", filter_value="ENV")
                dims = 3
                for _nid, env_attrs in env_nodes:
                    dims = env_attrs.get("dims", 3)
                    break

            modules = self.g.get_nodes(filter_key="type", filter_value="MODULE")
            for mid, _module in modules:
                if "GHOST" in mid.upper():
                    continue

                methods = self.g.get_neighbor_list_rel(
                    trgt_rel="has_method",
                    node=mid,
                    as_dict=True,
                )
                if not methods:
                    continue

                fields = self.g.get_neighbor_list_rel(
                    trgt_rel="has_field",
                    node=mid,
                    as_dict=True,
                )
                if not fields:
                    continue

                # Collect all required keys from methods (params + return_key)
                required_keys = set()
                for _eqid, eqattrs in methods.items():
                    params = eqattrs.get("params", [])
                    if isinstance(params, str):
                        try:
                            params = json.loads(params) if params else []
                        except json.JSONDecodeError:
                            params = []
                    return_key = eqattrs.get("return_key")
                    if return_key:
                        required_keys.add(return_key)
                    for p in params:
                        if p:
                            required_keys.add(p)

                # For each field, add missing keys with default value and axis_def 0
                for fid, fattrs in fields.items():
                    keys = fattrs.get("keys", [])
                    if isinstance(keys, str):
                        try:
                            keys = json.loads(keys) if keys else []
                        except json.JSONDecodeError:
                            keys = []
                    values = fattrs.get("values", [])
                    if isinstance(values, str):
                        try:
                            values = json.loads(values) if values else []
                        except json.JSONDecodeError:
                            values = []

                    axis_def = fattrs.get("axis_def", [])
                    if isinstance(axis_def, str):
                        try:
                            axis_def = json.loads(axis_def) if axis_def else []
                        except json.JSONDecodeError:
                            axis_def = []
                    # Pad axis_def to match keys length if needed
                    while len(axis_def) < len(keys):
                        axis_def.append(None)

                    missing_keys = [k for k in required_keys if k not in keys]
                    if not missing_keys:
                        continue

                    default_val = [0 for _ in range(dims)]
                    for key in missing_keys:
                        keys.append(key)
                        values.append(default_val)
                        axis_def.append(0)

                    self.g.update_node({
                        "id": fid,
                        "keys": keys,
                        "values": values,
                        "axis_def": axis_def,
                    })
        except Exception as e:
            print("Err sync_field_keys_from_methods", e)
        print("sync_field_keys_from_methods... done")

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
    # DATA STRUCT: modules=list of module IDs, injections=top-level {FIELD_ID: {pos_str: inj_id}}
    payload = {
        "type": "START_SIM",
        "data": {
            "config": {
                "env_814afeee6e234237b94b653af836b664": {
                    "modules": ["phi", "fermion", "HIGGS", "FERMION", "GAUGE"],
                    "injections": {
                        "PHOTON": {"[0,0,0]": "1", "[13,12,17]": "1"}
                    },
                }
            },
            "auth": {"session_id": 339617269692277, "user_id": "72b74d5214564004a3a86f441a4a112f"},
            "timestamp": "2026-02-20T07:59:47.867Z",
        }
    }
    env_id = "env_814afeee6e234237b94b653af836b664"
    user_id = "72b74d5214564004a3a86f441a4a112f"
    env_data = payload["data"]["config"][env_id]

    _qb = get_qbrain_table_manager()
    _qb.initialize_all_tables()
    params_manager = ParamsManager(_qb)
    env_manager = EnvManager(_qb)

    # Ensure env exists so model_path and animation_path can be saved
    existing = env_manager.retrieve_env_from_id(env_id)
    if not existing.get("envs"):
        env_manager.set_env(
            {"id": env_id, "sim_time": 100, "cluster_dim": 3, "dims": 3},
            user_id=user_id,
        )
        print(f"[guard __main__] created env {env_id}")

    grid_animation_recorder = None
    if os.getenv("GRID_STREAM_ENABLED", "false").lower() in ("true", "1"):
        try:
            from jax_test.grid.animation_recorder import GridAnimationRecorder
            env_cfg = {**{"dims": 3, "amount_of_nodes": 1}}
            grid_animation_recorder = GridAnimationRecorder(
                env_id=env_id,
                user_id=user_id,
                env_cfg=env_cfg,
                cfg={},
                env_manager=env_manager,
            )
        except Exception as e:
            print(f"[guard __main__] animation recorder init: {e}")

    g = GUtils()
    qfu = QFUtils(g)
    guard = Guard(
        qfu,
        g,
        user_id=user_id,
        injection_manager=InjectionManager(_qb),
        module_db_manager=ModuleWsManager(_qb),
        field_manager=FieldsManager(_qb),
        method_manager=MethodManager(_qb, params_manager=params_manager),
        params_manager=params_manager,
        env_manager=env_manager,
    )
    guard.main(env_id=env_id, env_data=env_data, grid_animation_recorder=grid_animation_recorder)

    # Debug: print saved paths from envs
    updated = env_manager.retrieve_env_from_id(env_id)
    if updated.get("envs"):
        row = updated["envs"][0]
        print(f"[guard __main__] model_path: {row.get('model_path', 'N/A')}")
        print(f"[guard __main__] animation_path: {row.get('animation_path', 'N/A')}")



async def run_start_sim_via_ws():
    user_id = payload.get("data", {}).get("auth", {}).get("user_id", "public")
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

equation = eqattrs.get("equation")
print("eq extracted", equation)

# EXTRACT PARAMS & OPERATORS
# -> method indices to all ops functions (VARIATION_INDICES is indexed by [module_idx][eq_idx])
raw_store = VARIATION_INDICES[midx][i] if midx < len(VARIATION_INDICES) and i < len(VARIATION_INDICES[midx]) else None
eq_store = list(raw_store.keys()) if isinstance(raw_store, dict) else (raw_store if isinstance(raw_store, list) else (list(params) if params else []))
method_indices: list[int] = eq_extractor_main(
    equation,
    eq_store_item=eq_store,
    
    
    

# Iterate Equations
                for i, (eqid, eqattrs) in enumerate(methods.items()):
                    # PARAMS from METHOD
                    params = eqattrs.get("params", [])

                    if isinstance(params, str):
                        params = json.loads(params)

                    params_origin = eqattrs.get("origin", [])
                    if isinstance(params_origin, str):
                        params_origin = json.loads(params_origin)
                    if not params_origin:
                        params_origin = [""] * len(params)
                    if params_origin is None:
                        params_origin = [""] * len(params)

                    # neighbor_vals: params whose origin is neighbor or interactant
                    neighbor_vals = [
                        params[pidx]
                        for pidx in range(len(params))
                        if pidx < len(params_origin) and params_origin[pidx] in ["neighbor", "interactant"]
                    ] if params else []

                    # params index for each item of neighbor_vals
                    neighbor_ctlr_entry = (
                        [params.index(nv) for nv in neighbor_vals]
                        if neighbor_vals else None
                    )

                    # param index for params of different origins
                    method_struct["NEIGHBOR_CTLR"][midx].append(neighbor_ctlr_entry)

                    method_struct["METHOD_PARAM_LEN_CTLR"][midx].append(
                        len(params)
                    )

# Field's own param
is_prefixed = pid.endswith("_")
is_self_prefixed = pid.startswith("_")
id_prev_val = pid.startswith("prev")
EXCLUDED_ORIGINS = ["neighbor", "interactant"]
)"""

import logging
import networkx as nx
from datetime import datetime
from typing import List, Dict, Tuple

from core.method_manager import MethodManager
from core.param_manager.params_lib import ParamsManager
from qf_utils.qf_utils import QFUtils
from qf_utils.field_utils import FieldUtils
from core.module_manager.mcreator import ModuleCreator
from core.module_manager.ws_modules_manager.modules_lib import module_manager, generate_numeric_id
from core.fields_manager.fields_lib import fields_manager
from core.user_manager.user import UserManager
from core.qbrain_manager import QBrainTableManager

class SMManager:
    def __init__(self):
        self.module_db_manager = module_manager
        self.field_manager = fields_manager

        # Managers
        self.user_manager = UserManager()
        self.param_manager = ParamsManager()
        self.method_manager = MethodManager()
        self.qb = QBrainTableManager()

    def check_sm_exists(self, user_id: str = "public"):
        print("check_sm_exists...")
        try:
            user=self.qb.row_from_id(user_id, table="users")
            if not user:
                return False
            user=user[0]
            if user["sm_stack_status"] == "created":
                print("check_sm_exists... done")
                return True
        except Exception as e:
            print("Err check_sm_exists:", e)
        print("check_sm_exists... done")
        return False


    def main(self, user_id: str = "public"):
        """
        Upsert standard nodes and edges from QFUtils to BigQuery tables.
        """
        print("PROCESSING STANDARD MANAGER WORKFLOW")

        """if self.check_sm_exists() is True:
            print(f"USERS {user_id} SM stack already enabled...")
            return"""

        # Create module stack
        qf = self._initialize_graph()

        # Upsert Nodes and Edges
        self._upsert_graph_content(qf, user_id)

        self.user_manager.set_standard_stack(user_id)
        print("FINISHED SM WORKFLOW")

    def _check_standard_stack(self, user_id: str) -> bool:
        """Check if standard stack exists for the user."""
        return self.user_manager.get_standard_stack(user_id)

    def enable_sm(self, user_id: str, session_id: str, env_id: str):
        """
        Link environment to Standard Model modules and fields.

        Return Structure:
        {
            "sessions": {
                session_id: {
                    "envs": {
                        env_id: {
                            "modules": {
                                module_id: {
                                    "fields": [field_id, ...]
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        print(f"ENABLING SM FOR Env: {env_id}, Session: {session_id}")
        fu = FieldUtils()
        sm_modules = fu.modules_fields
        
        env_module_links = []
        module_field_links = []
        
        now = datetime.now().isoformat()
        
        for module_name, fields in sm_modules.items():
            mid = module_name
            
            # Link Env -> Module
            env_module_links.append({
                "id": generate_numeric_id(),
                "env_id": env_id,
                "module_id": mid,
                "session_id": session_id,
                "user_id": user_id,
                "status": "active",
                "created_at": now,
                "updated_at": now
            })
            
            for field_name in fields:
                fid = field_name
                # Link Module -> Field
                module_field_links.append({
                    "id": generate_numeric_id(),
                    "module_id": mid,
                    "field_id": fid,
                    "session_id": session_id,
                    "env_id": env_id,
                    "user_id": user_id,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now
                })
        
        # Upsert
        if env_module_links:
            for row in env_module_links:
                self.qb.set_item("envs_to_modules", row, keys={"id": row["id"]})

        #
        if module_field_links:
            for row in module_field_links:
                self.qb.set_item("modules_to_fields", row, keys={"id": row["id"]})

        #
        formatted_modules = {}
        for mid, fids in sm_modules.items():
            formatted_modules[mid] = {"fields": fids}
         
        return {
            "sessions": {
                session_id: {
                    "envs": {
                        env_id: {
                            "modules": formatted_modules
                        }
                    }
                }
            }
        }

    def _initialize_graph(self) -> QFUtils:
        """Initialize QFUtils and load standard modules."""
        # Create separate graph for SM loading
        print("_initialize_graph...")
        qf = QFUtils(
            G=nx.Graph()
        )

        qf.build_interacion_G()

        qf.build_parameter()

        module_creator = ModuleCreator(
            G=qf.g.G,
            qfu=qf,
        )

        module_creator.load_sm()

        print("_initialize_graph finshed")
        return qf


    def _upsert_graph_content(self, qf: QFUtils, user_id: str):
        """Upsert nodes and edges from the graph to BigQuery."""
        # 1. Upsert Nodes
        logging.info("Upserting standard nodes...")
        modules, fields, params, methods = self._extract_nodes(qf, user_id)

        if modules:
            self.module_db_manager.set_module(modules, user_id)

        if fields:
            self.field_manager.set_field(fields, user_id)

        if params:
            self.param_manager.set_param(params, user_id)

        if methods:
            self.method_manager.set_method(methods, user_id)

        # 2. Upsert Edges
        logging.info("Upserting standard edges...")

    def _extract_nodes(
            self,
            qf: QFUtils,
            user_id: str
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Extract module and field nodes from the graph."""
        modules = []
        param_rows = []
        method_rows = []
        field_rows = []

        for nid, attrs in qf.g.G.nodes(data=True):
            ntype = attrs.get("type")

            if ntype == "MODULE":
                """# Get PARAM neighbors
                param_ids = qf.g.get_neighbor_list_rel(
                    node=nid,
                    trgt_rel="requires_param",
                    as_dict=True
                )
                #print(f"{nid} requires_param: {param_ids}")
                param_ids = list(param_ids.keys())"""


                # Get METHOD neighbors
                methods = qf.g.get_neighbor_list_rel(
                    node=nid,
                    trgt_rel="has_method",
                    as_dict=True
                )
                method_ids = list(methods.keys())
                print(f"{nid} has method ids: {method_ids}")

                # Get FIELD neighbors
                fields = qf.g.get_neighbor_list_rel(
                    node=nid,
                    trgt_rel="has_field",
                    as_dict=True
                )
                field_ids = list(fields.keys())
                print(f"{nid} has_field ids: {field_ids}")

                # Upsert Module
                module_data = {
                    "id": attrs.get("nid", nid),
                    "file_type": None,
                    "binary_data": None,
                    "user_id": user_id,
                    "status": "active",
                    "fields": field_ids,
                    "methods": method_ids,
                    **{
                        k: v
                        for k, v in attrs.items() if k not in [
                            "type",
                            "nid",
                            "params",
                            "code",
                        ]
                    },
                }
                modules.append(module_data)

            elif ntype == "FIELD":
                # Upsert Field
                keys = attrs.get("keys") or attrs.get("field_keys")
                values = attrs.get("values") or attrs.get("value")
                axis_def = attrs.get("axis_def")
                interactant_fields = attrs.get("interactant_fields")

                if keys is None or values is None:
                    print(f"Skipping malformed FIELD node {attrs}. Keys/Values missing.")
                    continue

                # Ensure list types for JSON serialization
                if keys is not None and not isinstance(keys, list):
                    keys = list(keys)
                if values is not None and not isinstance(values, list):
                    values = list(values)
                if axis_def is not None and not isinstance(axis_def, list):
                    axis_def = list(axis_def)

                field_data = {
                    "id": nid,
                    "keys": keys,
                    "values": values,
                    "axis_def": axis_def,
                    "module_id": attrs.get(
                        "module_id",
                        attrs["parent"][0]
                    ),
                    "interactant_fields": interactant_fields,
                }
                field_rows.append(field_data)

            elif ntype == "PARAM":
                param_data = {
                    "id": nid,
                    "type": attrs.get("type"),
                    "description": "",
                    "origin": "SM",
                    "const": attrs.get("const"),
                    "axis_def": attrs.get("axis_def"),
                    "value": attrs.get("value"),
                }
                param_rows.append(param_data)

            elif ntype == "METHOD":
                #
                param_ids = qf.g.get_neighbor_list_rel(
                    node=nid,
                    trgt_rel="requires_param",
                    as_dict=True
                )
                param_ids = list(param_ids.keys())

                method_data = {
                    "id": nid,
                    'return_key': attrs.get("return_key"),
                    "params": param_ids,
                    "jax_code": attrs.get("jax_code", attrs.get("code", None)),
                    "code": attrs.get("code", None),
                    "axis_def": attrs.get("code", None),
                }
                method_rows.append(method_data)
        
        return modules, field_rows, param_rows, method_rows

    def _extract_edges(self, qf: QFUtils, user_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract edges from the graph."""
        mfs = []
        ffs = []

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
        
        return mfs, ffs

# Instantiate
sm_manager = SMManager()

def handle_enable_sm(payload):
    print("handle_enable_sm...")
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    env_id = auth.get("env_id")

    if not all([user_id, session_id, env_id]):
         return {"error": "Missing param for ENABLE_SM"}

    # Call SM Manager to link SM modules to env
    res = sm_manager.enable_sm(user_id, session_id, env_id)

    # Returns {envs:[], fields:[]}
    return {
        "type": "ENABLE_SM",
        "data": res
    }

import base64
import json
import logging
import random
from typing import Dict, Any, List, Optional

from google.cloud import bigquery
from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from core.method_manager.gen_type import generate_methods_out_schema
from core.method_manager.method_processor import MethodDataProcessor
from core.method_manager.xtrct_prompt import xtrct_method_prompt
from core.param_manager.params_lib import ParamsManager
from core.qbrain_manager import QBrainTableManager
from qf_utils.qf_utils import QFUtils

# Define Schemas
METHOD_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("code", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("description", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("params", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("equation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("jax_code", "STRING", mode="NULLABLE"),
]

SESSIONS_METHODS_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("method_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
]

def generate_numeric_id() -> str:
    """Generate a random numeric ID."""
    return str(random.randint(1000000000, 9999999999))

class MethodManager(BQCore, MethodDataProcessor):
    DATASET_ID = "QBRAIN"
    METHODS_TABLE = "methods"
    SESSIONS_METHODS_TABLE = "sessions_to_methods"

    def __init__(self):
        super().__init__(dataset_id=self.DATASET_ID)
        self.qb = QBrainTableManager()
        self.table_ref = f"{self.METHODS_TABLE}"
        self.session_link_ref = f"{self.SESSIONS_METHODS_TABLE}"

        self.qfu = QFUtils()

        self.param_manager = ParamsManager()
        self._extract_prompt = None  # built lazily to avoid circular import with case
        self._ensure_method_table()

    def _ensure_method_table(self):
        """Check if methods table exists, create if not."""
        table_ref = f"{self.pid}.{self.DATASET_ID}.{self.METHODS_TABLE}"
        try:
            self.bqclient.get_table(table_ref)
            logging.info(f"Table {table_ref} exists.")
            # Ensure new columns exist
            self.insert_col(self.METHODS_TABLE, "equation", "STRING")
            self.insert_col(self.METHODS_TABLE, "jax_code", "STRING")
        except Exception as e:
            print(f"Error ensuring method table: {e}")
            logging.info(f"Creating table {table_ref}.")
            table = bigquery.Table(table_ref, schema=METHOD_SCHEMA)
            self.bqclient.create_table(table)
            logging.info(f"Table {table_ref} created.")

    def extract_from_file_bytes(
            self,
            content: bytes or str,
            params,
            fallback_params,
            instructions,

    ) -> Optional[Dict[str, Any]]:
        """
        Extract manager-specific method/equation content from file bytes using the static prompt.
        Uses Gem LLM with req_struct/out_struct from SET_METHOD case.
        """
        try:
            from gem_core.gem import Gem
            gem = Gem()

            prompt = xtrct_method_prompt(
                params,
                fallback_params,
                instructions,
            )

            try:
                content = f"{prompt}\n\n--- FILE CONTENT ---\n{content}"
                response = gem.ask(
                    content,
                    config=generate_methods_out_schema
                )
                text = (response or "").strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(text)
                if "methods" in parsed:
                    return {
                        "methods": parsed["methods"] if isinstance(parsed["methods"], list) else [parsed["methods"]]}
                if "id" in parsed or "equation" in parsed:
                    return {"methods": parsed}

                print("extracted mehods:", parsed)
                return {"methods": parsed}
            except Exception as e:
                print("Err method amanger extract_from_file_bytes", e)
        except Exception as e:
            logging.error(f"MethodManager extract_from_file_bytes error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def set_method(self, rows: List[Dict] or Dict, user_id: str):
        if isinstance(rows, dict):
            rows = [rows]

        for row in rows:
            row["user_id"] = user_id

            param_data = self.qb.row_from_id(
                row["params"],
                table="params",
            )

            # SET AXIS FOR METHOD
            row["axis_def"] = [
                p.get("axis_def", None)
                for p in param_data
            ]

        print("set method rows", len(rows))
        self.qb.set_item(self.METHODS_TABLE, rows)

    def link_session_method(self, session_id: str, method_id: str, user_id: str):
        """Link a method to a session."""
        row_id = generate_numeric_id()
        row = {
            "id": row_id,
            "session_id": session_id,
            "method_id": method_id,
            "user_id": user_id,
            "status": "active"
        }

        self.qb.set_item(self.SESSIONS_METHODS_TABLE, row)

    def delete_method(self, method_id: str, user_id: str):
        """Delete method and its links."""
        # Delete from methods (Soft Delete)
        self.qb.del_entry(
            nid=method_id,
            table=self.METHODS_TABLE,
            user_id=user_id
        )

        # Delete from sessions_to_methods (Soft Delete)
        query2 = f"""
            UPDATE `{self.pid}.{self.DATASET_ID}.{self.SESSIONS_METHODS_TABLE}`
            SET status = 'deleted'
            WHERE method_id = @method_id AND user_id = @user_id
        """

        job_config2 = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("method_id", "STRING", method_id),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        self.run_query(query2, job_config=job_config2, conv_to_dict=True)

    def rm_link_session_method(self, session_id: str, method_id: str, user_id: str):
        """Remove link session method (soft delete)."""
        self.qb.rm_link_session_link(
            session_id=session_id,
            nid=method_id,
            user_id=user_id,
            session_link_table=self.session_link_ref,
            session_to_link_name_id="method_id"
        )

    def update_method_params(self, method_id: str, user_id: str, params: Dict[str, Any] = None):
        """
        Update params field of a method.
        """
        if params is None:
            params = {}

        self.qb.set_item(
            self.METHODS_TABLE, 
            {"params": params}, 
            keys={"id": method_id, "user_id": user_id}
        )

    def retrieve_user_methods(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all methods for a user."""
        result = self.qb.get_users_entries(
            user_id=user_id,
            table=self.table_ref
        )
        return [dict(row) for row in result]

    def retrieve_session_methods(self, session_id: str, user_id: str, select: str = "*") -> List[Dict[str, Any]]:
        """Retrieve methods for a session."""
        # get sessions_to_methods rows -> get method rows
        links = self.qb.list_session_entries(
            user_id=user_id,
            session_id=session_id,
            table=self.session_link_ref,
            select="method_id"
        )
        
        method_ids = [row['method_id'] for row in links]
        
        result = self.qb.row_from_id(
            nid=method_ids,
            select="*",
            table=self.table_ref
        )

        return [dict(row) for row in result]

    def get_method_by_id(
            self, method_id: str or list, select: str = "*") -> Optional[Dict[str, Any]]:
        """Get a single method by ID."""
        if isinstance(method_id, str):
            method_id = [method_id]

        rows = self.qb.row_from_id(
            nid=method_id,
            select=select,
            table=self.METHODS_TABLE
        )
        return {"methods": rows}


# Instantiate
method_manager = MethodManager()

# -- RELAY HANDLERS --

def handle_list_users_methods(payload):
    """
    receive "LIST_USERS_METHODS": auth=user_id:str -> SEND_USERS_METHODS
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    if not user_id:
        return {"error": "Missing user_id"}
    
    methods = method_manager.retrieve_user_methods(user_id)
    
    return {
        "type": "LIST_USERS_METHODS", 
        "data": {"methods": methods} 
    }

def handle_send_sessions_methods(payload):
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    
    if not user_id or not session_id:
        return {"error": "Missing user_id or session_id"}

    methods = method_manager.retrieve_session_methods(session_id, user_id)
    return {
        "type": "GET_SESSIONS_METHODS",
        "data": {"methods": methods}
    }

def handle_get_sessions_methods(payload):
    return handle_send_sessions_methods(payload)

def handle_link_session_method(payload):
    """
    receive "LINK_SESSION_METHOD": auth={user_id:str, method_id:str, session_id:str}
    -> LINK_SESSION_METHOD -> SEND_SESSIONS_METHODS
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    method_id = auth.get("method_id")
    session_id = auth.get("session_id")
    
    if not all([user_id, method_id, session_id]):
        return {"error": "Missing required auth fields"}
        
    method_manager.link_session_method(session_id, method_id, user_id)
    return handle_send_sessions_methods(payload)

def handle_rm_link_session_method(payload):
    """
    receive type="RM_LINK_SESSION_METHOD", auth={user_id:str, session_id:str, method_id:str}
    -> update sessions_to_methods-table row with status = deleted -> SEND_SESSION_METHODS
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    method_id = auth.get("method_id")
    
    if not all([user_id, session_id, method_id]):
        return {"error": "Missing required auth fields"}
        
    method_manager.rm_link_session_method(session_id, method_id, user_id)
    return handle_send_sessions_methods(payload)

def handle_del_method(payload):
    """
    receive "DEL_METHOD". auh={method_id:str, user_id:str}
    -> delete -> SEND_USERS_METHODS
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    method_id = auth.get("method_id")
    
    if not user_id or not method_id:
        return {"error": "Missing user_id or method_id"}
        
    method_manager.delete_method(method_id, user_id)
    return handle_list_users_methods(payload)

def handle_set_method(payload):
    """
    receive "SET_METHOD": data={description, equation, id, params}, auth={user_id}
    -> generate jax_code -> insert -> SEND_USERS_METHODS
    """
    auth = payload.get("auth", {})
    data = payload.get("data", {})
    user_id = auth.get("user_id")
    
    # Data is flat in the new payload
    method_data = data
    
    # Extract specific fields
    equation = method_data.get("equation")
    params = method_data.get("params") # List of strings
    
    if not user_id:
        return {"error": "Missing user_id"}

    original_id = auth.get("original_id")
    if original_id:
        try:
            method_manager.delete_method(original_id, user_id)
        except:
            pass

    method_data["code"] = equation


    # generat jax code
    if "jax_code" not in  method_data:
        from core.file_manager.file_lib import file_manager
        method_data["jax_code"] = file_manager.jax_predator(method_data["code"])

    # Ensure ID
    if "id" not in method_data or not method_data["id"]:
        method_data["id"] = generate_numeric_id()
        
    # Process Params if list
    if isinstance(params, list):
         origins = method_data.get("origins")
         if isinstance(origins, list):
             key_counts = {}
             for i, key in enumerate(params):
                 if i < len(origins):
                     origin = origins[i]
                     if key in key_counts:
                         if origin != "self":
                             params[i] = "_" * key_counts[key] + key
                     key_counts[key] = key_counts.get(key, 0) + 1
         method_data["params"] = json.dumps(params)
         
    # Generate JAX Code
    if equation:
        print(f"Generating JAX code for equation: {equation}")
        try:
            from core.file_manager.file_lib import file_manager
            jax_code = file_manager.jax_predator(equation)
            method_data["jax_code"] = jax_code
            print("JAX Code generated:", jax_code)
        except Exception as e:
            print(f"Failed to generate JAX code: {e}")

    method_manager.set_method(method_data, user_id)
    return handle_list_users_methods(payload)

def handle_get_method(payload):
    """
    receive "GET_METHOD": auth={method_id:str}
    -> get row -> return {type: "GET_METHOD", data: ...}
    """
    auth = payload.get("auth", {})
    method_id = auth.get("method_id")
    
    if not method_id:
        return {"error": "Missing method_id"}
        
    row = method_manager.get_method_by_id(method_id)
    if not row:
        return {"error": "Method not found"}

    return {
        "type": "GET_METHOD",
        "data": row
    }

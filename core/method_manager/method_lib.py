import base64
import json
import logging
import random
from typing import Dict, Any, List, Optional, Callable

import numpy as np
from google.cloud import bigquery
from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from core.method_manager.gen_type import generate_methods_out_schema
from core.method_manager.xtrct_prompt import xtrct_method_prompt
from core.module_manager.create_runnable import create_runnable
from core.qbrain_manager import get_qbrain_table_manager, QBrainTableManager
from core.handler_utils import require_param, require_param_truthy, get_val
from qf_utils.qf_utils import QFUtils
# 
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

class MethodManager:
    DATASET_ID = "QBRAIN"
    METHODS_TABLE = "methods"
    SESSIONS_METHODS_TABLE = "sessions_to_methods"

    def __init__(self, qb:QBrainTableManager, params_manager):
        self.qb = qb
        self.params_manager=params_manager
        self.pid = qb.pid
        self.bqclient = qb.bqclient
        self.insert_col = qb.insert_col
        self.table_ref = f"{self.METHODS_TABLE}"
        self.session_link_ref = f"{self.SESSIONS_METHODS_TABLE}"
        self.qfu = QFUtils()
        self._extract_prompt = None  # built lazily to avoid circular import with case

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


    def execute_method_testwise(self, methods:list[dict]):
        return_key_param_entries = self.qb.row_from_id([m["return_key"] for m in methods.values()], table="params")
        param_entries = self.qb.row_from_id(list(set([m["params"] for m in methods.values()])), table="params")
        param_entries = {p["id"]: p for p in param_entries}
        adapted_return_params = []

        for i, _def_content in enumerate(methods):
            params = _def_content.get("params")
            equation = _def_content.get("equation")
            return_key = _def_content.get("return_key")
            code = _def_content.get("code")
            _def_id = _def_content["id"]

            # try get param shape from db
            param_shape=None
            param_entry = return_key_param_entries[i]
            if param_entry:
                param_shape = param_entry["shape"]

            if not param_shape or not param_shape:
                # need calc testwise to infer
                # collect param values (shape? type?)
                val_params = []
                for p_key in params:
                    is_array = p_key["shape"] and isinstance(p_key, (list, tuple)) and len(p_key)
                    if is_array:
                        val_params.append(np.ones(param_entries[p_key]["shape"]))
                    else:
                        val_params.append(1)

                runnable: Callable = create_runnable(code)
                result = runnable(*params)
                if result:
                    result_shape = np.array(result).shape

                    # upsert resul
                    if param_entry:
                        payload = param_entry

                    else:
                        payload = dict(
                            id=return_key,
                            param_type=type(result),
                            axis_def=0,
                            description=f"return key of {_def_id}"
                        )
                    payload["shape"]=result_shape
                    adapted_return_params.append(payload)

                else:
                    print("no result shape found...")
            else:
                raise Exception(f"runnable failed for code {code}, params {params},")

            self.params_manager.set_param(
                param_data=adapted_return_params
            )
            print("eq extractted", equation)










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
_default_bqcore = BQCore(dataset_id="QBRAIN")
_default_method_manager = MethodManager(get_qbrain_table_manager(_default_bqcore))
method_manager = _default_method_manager  # backward compat

# -- RELAY HANDLERS --

def handle_list_users_methods(data=None, auth=None):
    """Retrieve all methods owned by a user. Required: user_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    if err := require_param(user_id, "user_id"):
        return err
    from core.managers_context import get_method_manager
    return {"type": "LIST_USERS_METHODS", "data": {"methods": get_method_manager().retrieve_user_methods(user_id)}}


def handle_send_sessions_methods(data=None, auth=None):
    """Retrieve all methods linked to a session. Required: user_id, session_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    from core.managers_context import get_method_manager
    return {"type": "GET_SESSIONS_METHODS", "data": {"methods": get_method_manager().retrieve_session_methods(session_id, user_id)}}


def handle_get_sessions_methods(data=None, auth=None):
    """Alias for handle_send_sessions_methods."""
    return handle_send_sessions_methods(data=data, auth=auth)


def handle_link_session_method(data=None, auth=None):
    """Link a method to a session. Required: user_id, method_id, session_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    method_id = get_val(data, auth, "method_id")
    session_id = get_val(data, auth, "session_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(method_id, "method_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    from core.managers_context import get_method_manager
    get_method_manager().link_session_method(session_id, method_id, user_id)
    return handle_send_sessions_methods(data={"session_id": session_id}, auth={"user_id": user_id})


def handle_rm_link_session_method(data=None, auth=None):
    """Remove the link between a session and a method. Required: user_id, session_id, method_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    method_id = get_val(data, auth, "method_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    if err := require_param(method_id, "method_id"):
        return err
    from core.managers_context import get_method_manager
    get_method_manager().rm_link_session_method(session_id, method_id, user_id)
    return handle_send_sessions_methods(data={"session_id": session_id}, auth={"user_id": user_id})


def handle_del_method(data=None, auth=None):
    """Delete a method by ID. Required: user_id, method_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    method_id = get_val(data, auth, "method_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(method_id, "method_id"):
        return err
    from core.managers_context import get_method_manager
    get_method_manager().delete_method(method_id, user_id)
    return handle_list_users_methods(data={}, auth={"user_id": user_id})


def handle_set_method(data=None, auth=None):
    """Create or update a method. Required: user_id (auth), data (method dict with equation, description, id, params). Optional: original_id (auth)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    original_id = get_val(data, auth, "original_id")
    method_data = data if isinstance(data, dict) else None
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param_truthy(method_data, "data"):
        return err

    equation = method_data.get("equation")
    params = method_data.get("params")

    method_data["code"] = equation
    if "jax_code" not in method_data:
        from core.managers_context import get_file_manager
        method_data["jax_code"] = get_file_manager().jax_predator(method_data["code"])

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
            from core.managers_context import get_file_manager
            jax_code = get_file_manager().jax_predator(equation)
            #method_data["jax_code"] = jax_code
            print("JAX Code generated:", jax_code)
        except Exception as e:
            print(f"Failed to generate JAX code: {e}")

    # handle set method
    from core.managers_context import get_method_manager
    mm = get_method_manager()
    # test_exec_method

    mm.execute_method_testwise([method_data])

    # set
    mm.set_method(method_data, user_id)
    return handle_list_users_methods(data={}, auth={"user_id": user_id})


def handle_get_method(data=None, auth=None):
    """Retrieve a single method by ID. Required: method_id (auth or data)."""
    data, auth = data or {}, auth or {}
    method_id = get_val(data, auth, "method_id")
    if err := require_param(method_id, "method_id"):
        return err
    from core.managers_context import get_method_manager
    row = get_method_manager().get_method_by_id(method_id)
    if not row:
        return {"error": "Method not found"}
    return {"type": "GET_METHOD", "data": row}

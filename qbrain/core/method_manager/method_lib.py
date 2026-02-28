import json
import logging
import random
from typing import Dict, Any, List, Optional, Callable, Tuple, Union

import numpy as np
from google.cloud import bigquery
from qbrain.a_b_c.bq_agent._bq_core.bq_handler import BQCore
from qbrain.core.method_manager.gen_type import generate_methods_out_schema
from qbrain.core.method_manager.xtrct_prompt import xtrct_method_prompt
from qbrain.core.module_manager.create_runnable import create_runnable
from qbrain.core.param_manager.params_lib import ParamsManager
from qbrain.core.qbrain_manager import get_qbrain_table_manager, QBrainTableManager
from qbrain.core.handler_utils import require_param, require_param_truthy, get_val
from qbrain.qf_utils.qf_utils import QFUtils



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
        self.table_ref = f"{self.METHODS_TABLE}"
        self.session_link_ref = f"{self.SESSIONS_METHODS_TABLE}"
        self.qfu = QFUtils()
        self._extract_prompt = None  # built lazily to avoid circular import with case

    def _ensure_method_table(self):
        schema = {f.name: f.field_type for f in METHOD_SCHEMA}
        self.qb.get_table_schema(table_id=self.METHODS_TABLE, schema=schema, create_if_not_exists=True)
        self.qb.insert_col(self.METHODS_TABLE, "equation", "STRING")
        self.qb.insert_col(self.METHODS_TABLE, "jax_code", "STRING")


    def execute_method_testwise(self, methods:list[dict], user_id, g):
        """
        Collect shapes from test execution with param vals from param shape
        """
        print("execute_method_testwise...")

        test_dims = 3
        return_key_ids = [m["return_key"] for m in methods]
        print("execute_method_testwise return_key_ids", return_key_ids)
        return_key_param_entries = self.qb.row_from_id(return_key_ids, table="params")

        if return_key_param_entries:
            return_key_param_entries = {p["id"]: p for p in return_key_param_entries}

        param_entries = self.qb.get_users_entries(user_id, table="params")
        if param_entries:
            param_entries = {p["id"]: p for p in param_entries}

        for k in return_key_ids:
            v = {}
            if g.G.has_node(k):
                v = g.G.nodes[k]
            if return_key_param_entries and k in return_key_param_entries:
                return_key_param_entries[k].update(v)
            else:
                return_key_param_entries[k] = v

        adapted_return_params = []

        for i, _def_content in enumerate(methods):
            params = _def_content.get("params")
            equation = _def_content.get("equation")
            return_key = _def_content.get("return_key")
            code = _def_content.get("code")
            try:

                _def_id = _def_content["id"]

                # try get param shape from db
                param_shape=None
                param_entry = return_key_param_entries[return_key]
                if param_entry:
                    param_shape = param_entry["shape"]

                if not param_shape:
                    # CALC THE RESULT SHAPE
                    val_params = []
                    for p_key in params:
                        p_shape = param_entries[p_key]["shape"]
                        param_type = param_entries[p_key]["param_type"]
                        param_value = param_entries[p_key]["value"]

                        # adapt_to_n_dims returns nested data or scalar; used to infer if param is array-like
                        resolved = self.adapt_to_n_dims(
                            p_key=p_key,
                            param_type=param_type,
                            flat_value=param_value,
                            shape=p_shape,
                        )

                        is_array = resolved and isinstance(resolved, (list, tuple)) and len(resolved)
                        # Build placeholder: use complex64 so physics methods (e.g. calc_psi_bar) that
                        # call .conj() on args do not fail (Python int has no .conj())
                        if is_array:
                            arr_shape = np.array(resolved).shape
                            val_params.append(np.ones(arr_shape, dtype=np.complex64))
                        else:
                            val_params.append(np.asarray(1, dtype=np.complex64))
                    print("val_params", params, val_params)

                    runnable: Callable = create_runnable(code)
                    result = runnable(*val_params)

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
                    param_data=adapted_return_params,
                    user_id=user_id,
                )
                print("eq extractted", equation)
            except Exception as e:
                # Skip on shape mismatch / validation failure - continue workflow (minimal fix)
                print(f"Err method manager execute_method_testwise", e, params)
        print("execute_method_testwise... done")


    def adapt_to_n_dims(self, p_key, param_type: type, flat_value: list, shape: Tuple[int, ...]) -> Union[List, Tuple]:
        """
        Recursively nests a 1D list into N-dimensions based on the provided shape.

        :param param_type: The desired container type (list or tuple)
        :param flat_value: The 1D data source
        :param shape: A tuple defining the dimensions (e.g., (3, 2, 2))
        :return: Nested structure of param_type
        """
        try:
            if not shape:
                print("no shape for", p_key, param_type, flat_value, shape)
                return flat_value[0] if flat_value else None

            # Calculate how many elements belong in each sub-slice of the current dimension
            # Example: if shape is (3, 4) and flat_value has 12 items,
            # the first dimension (3) contains 3 groups of 4 items each.
            stride = 1
            for dim in shape[1:]:
                stride *= dim

            nested = []
            for i in range(0, len(flat_value), stride):
                chunk = flat_value[i: i + stride]

                # If there are more dimensions to process, recurse
                if len(shape) > 1:
                    nested.append(self.adapt_to_n_dims(param_type, chunk, shape[1:]))
                else:
                    # Base case: reached the last dimension
                    nested.extend(chunk)
                    break

            ptype = param_type(nested)
            print("param_shape", ptype)
            return ptype
        except Exception as e:
            print(f"Err method manager adapt_to_n_dims", e)
        return None



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

    def set_method(self, rows: List[Dict] or Dict, user_id: str, g=None):
        if isinstance(rows, dict):
            rows = [rows]

        if g is not None:
            self.execute_method_testwise(rows, user_id, g)



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
            id=method_id,
            table=self.METHODS_TABLE,
            user_id=user_id
        )

        # Delete from sessions_to_methods (Soft Delete)
        query2 = f"UPDATE {self.qb._table_ref(self.SESSIONS_METHODS_TABLE)} SET status = 'deleted' WHERE method_id = @method_id AND user_id = @user_id"
        self.qb.db.execute(query2, params={"method_id": method_id, "user_id": user_id})

    def rm_link_session_method(self, session_id: str, method_id: str, user_id: str):
        """Remove link session method (soft delete)."""
        self.qb.rm_link_session_link(
            session_id=session_id,
            id=method_id,
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
            id=method_ids,
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
            id=method_id,
            select=select,
            table=self.METHODS_TABLE
        )
        return {"methods": rows}








# Instantiate
_default_bqcore = BQCore(dataset_id="QBRAIN")
_qb:QBrainTableManager = get_qbrain_table_manager(_default_bqcore)
params_manager = ParamsManager(_qb)

_default_method_manager = MethodManager(_qb, params_manager)
method_manager = _default_method_manager  # backward compat

# -- RELAY HANDLERS --

def handle_list_users_methods(data=None, auth=None):
    """Retrieve all methods owned by a user. Required: user_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    if err := require_param(user_id, "user_id"):
        return err
    from qbrain.core.managers_context import get_method_manager
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
    from qbrain.core.managers_context import get_method_manager
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
    from qbrain.core.managers_context import get_method_manager
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
    from qbrain.core.managers_context import get_method_manager
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
    from qbrain.core.managers_context import get_method_manager
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
        from qbrain.core.managers_context import get_file_manager
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
            from qbrain.core.managers_context import get_file_manager
            jax_code = get_file_manager().jax_predator(equation)
            #method_data["jax_code"] = jax_code
            print("JAX Code generated:", jax_code)
        except Exception as e:
            print(f"Failed to generate JAX code: {e}")

    # handle set method
    from qbrain.core.managers_context import get_method_manager
    mm = get_method_manager()
    # test_exec_method



    # set
    mm.set_method(method_data, user_id)
    return handle_list_users_methods(data={}, auth={"user_id": user_id})


def handle_get_method(data=None, auth=None):
    """Retrieve a single method by ID. Required: method_id (auth or data)."""
    data, auth = data or {}, auth or {}
    method_id = get_val(data, auth, "method_id")
    if err := require_param(method_id, "method_id"):
        return err
    from qbrain.core.managers_context import get_method_manager
    row = get_method_manager().get_method_by_id(method_id)
    if not row:
        return {"error": "Method not found"}
    return {"type": "GET_METHOD", "data": row}

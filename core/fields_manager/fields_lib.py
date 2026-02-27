import base64
import random
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from google.cloud import bigquery
from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from core.fields_manager.prompt import extract_fields_prompt
from core.fields_manager.set_type import SetFieldItem

from core.qbrain_manager import get_qbrain_table_manager
from core.handler_utils import require_param, require_param_truthy, get_val
from qf_utils.all_subs import FERMIONS, G_FIELDS, H
from qf_utils.qf_utils import QFUtils

_FIELDS_DEBUG = "[FieldsManager]"

# Define Schemas
FIELD_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("params", "STRING", mode="NULLABLE"),  # Storing JSON as STRING
    bigquery.SchemaField("module", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
]

FIELD_TO_FIELD_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("field_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("interactant_field_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
]

MODULE_TO_FIELD_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("module_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("field_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
]

def generate_numeric_id() -> str:
    """Generate a random numeric ID."""
    return str(random.randint(1000000000, 9999999999))

class FieldsManager:
    DATASET_ID = "QBRAIN"
    FIELDS_TABLE = "fields"
    FIELDS_TO_FIELDS_TABLE = "fields_to_fields"
    MODULE_TO_FIELD_TABLE = "module_to_field"

    def __init__(self, qb):
        self.qb = qb
        self.pid = qb.pid
        self.table = f"{self.FIELDS_TABLE}"
        self.qfu = QFUtils()
        self._extract_prompt = None  # built lazily to avoid circular import with case



    def resturn_config(self):
        config = {
            "response_mime_type": "application/json",
            "response_schema": SetFieldItem
        }
        return config

    def _get_extract_prompt(self,  user_input, params,fallback_params) -> (str, str):
        """Build prompt lazily to avoid circular import with case."""
        prompt = extract_fields_prompt(
            instructions=user_input,
            params=params,
            fallback_params=fallback_params
        )
        respoonse_cfg = self.resturn_config()
        return prompt, respoonse_cfg

    def extract_from_file_bytes(
            self,
            file_bytes: bytes or str,
            user_input,
            params,
            fallback_params,

    ) -> Optional[Dict[str, Any]]:
        """
        Extract manager-specific field content from file bytes using the static prompt.
        Uses Gem LLM with req_struct/out_struct from SET_FIELD case.
        """
        try:
            from gem_core.gem import Gem
            gem = Gem()
            prompt, config = self._get_extract_prompt(
                user_input,
                params,
                fallback_params=fallback_params,
            )
            try:
                content = f"{prompt}\n\n--- FILE CONTENT ---\n{file_bytes}"
                response = gem.ask(content, config)
                text = (response or "").strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(text)
                if "field" in parsed:
                    return {"field": parsed["field"]}
                return {"field": parsed}
            except Exception as e:
                print("Err extract field content", e)

        except Exception as e:
            print(f"{_FIELDS_DEBUG} extract_from_file_bytes error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _ensure_fields_table(self):
        schema = {f.name: f.field_type for f in FIELD_SCHEMA}
        self.qb.get_table_schema(table_id=self.FIELDS_TABLE, schema=schema, create_if_not_exists=True)

    def _ensure_fields_to_fields_table(self):
        schema = {f.name: f.field_type for f in FIELD_TO_FIELD_SCHEMA}
        self.qb.get_table_schema(table_id=self.FIELDS_TO_FIELDS_TABLE, schema=schema, create_if_not_exists=True)

    def get_field_interactants(self, field_id: str, select: str = "*") -> Dict[str, list[str or None]]:
        try:
            query = f"SELECT {select} FROM {self.qb._table_ref(self.FIELDS_TO_FIELDS_TABLE)} WHERE field_id = @field_id AND (status != 'deleted' OR status IS NULL)"
            result = self.qb.db.run_query(query, conv_to_dict=True, params={"field_id": str(field_id)})
            interactant_ids = [item["interactant_field_id"] for item in result]
            return {"interactant_ids": interactant_ids}
        except Exception as e:
            print(f"{_FIELDS_DEBUG} get_field_interactants: error: {e}")
            import traceback
            traceback.print_exc()
            return {"interactant_ids": []}


    def _ensure_module_to_field_table(self):
        schema = {f.name: f.field_type for f in MODULE_TO_FIELD_SCHEMA}
        self.qb.get_table_schema(table_id=self.MODULE_TO_FIELD_TABLE, schema=schema, create_if_not_exists=True)


    def get_fields_by_user(self, user_id: str, select: str = "*") -> List[Dict[str, Any]]:
        try:
            print(f"{_FIELDS_DEBUG} get_fields_by_user: user_id={user_id}")
            result = self.qb.get_users_entries(
                user_id=user_id,
                table=self.table,
                select=select
            )
            fields = [dict(row) for row in result]
            print(f"{_FIELDS_DEBUG} get_fields_by_user: got {len(fields)} field(s)")
            return fields
        except Exception as e:
            print(f"{_FIELDS_DEBUG} get_fields_by_user: error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_fields_by_module(self, module_id: str, user_id: str, select: str = "f.*") -> List[Dict[str, Any]]:
        try:
            print(f"{_FIELDS_DEBUG} get_fields_by_module: module_id={module_id}, user_id={user_id}")
            module_linked_rows = self.qb.get_modules_linked_rows(
                module_id=module_id,
                user_id=user_id,
                table_name=self.MODULE_TO_FIELD_TABLE,
                linked_row_id="field_id",
                linked_row_id_name="field_id"
            )
            field_ids = [row["field_id"] for row in module_linked_rows]
            fields = self.qb.row_from_id(
                id=field_ids,
                select="*",
                table=self.table
            )
            for f in fields:
                if f.get("params") and isinstance(f["params"], str):
                    try:
                        f["params"] = json.loads(f["params"])
                    except Exception as e:
                        print(f"{_FIELDS_DEBUG} get_fields_by_module: parse params: {e}")
            print(f"{_FIELDS_DEBUG} get_fields_by_module: got {len(fields)} field(s)")
            return fields
        except Exception as e:
            print(f"{_FIELDS_DEBUG} get_fields_by_module: error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def delete_field(self, field_id: str, user_id: str):
        try:
            print(f"{_FIELDS_DEBUG} delete_field: field_id={field_id}, user_id={user_id}")
            self.qb.del_entry(
                id=field_id,
                table=self.table,
                user_id=user_id
            )
            print(f"{_FIELDS_DEBUG} delete_field: done")
        except Exception as e:
            print(f"{_FIELDS_DEBUG} delete_field: error: {e}")
            import traceback
            traceback.print_exc()
            raise
        

    def get_fields_by_id(self, field_ids: List[str], select: str = "*") -> List[Dict[str, Any]]:
        """Get fields by list of IDs."""
        result = self.qb.row_from_id(
            id=field_ids,
            select=select,
            table=self.table
        )
        fields = [dict(row) for row in result]
        return {"fields": fields}


    def set_field(
            self,
            field_data: Dict[str, Any] or List[dict],
            user_id: str
    ):
        print("set_field")
        if not isinstance(field_data, list):
            field_data = [field_data]

        now = datetime.now().isoformat()

        for f in field_data:
            f["user_id"] = user_id
            f["status"] = "active"
            f["created_at"] = now
            f["updated_at"] = now

        self.qb.set_item(
            self.FIELDS_TABLE,
            field_data,
        )
        print("set_field... done")


    def link_module_field(self, data:list):
        """
        {
            "id": generate_numeric_id(),
            "module_id": module_id,
            "field_id": field_id,
            "session_id": session_id,
            "env_id": env_id,
            "user_id": user_id
        }
        """
        print("link_module_field", len(data), "rows")
        try:
            if isinstance(data, dict):
                 data = [data]
            self.qb.set_item(self.MODULE_TO_FIELD_TABLE, data)
        except Exception as e:
            print(f"{_FIELDS_DEBUG} link_module_field: error: {e}")
            import traceback
            traceback.print_exc()
            raise


    def link_field_field(self, data:list):
        """Link two fields together.
        {
            "id": generate_numeric_id(),
            "field_id": field_id,
            "interactant_field_id": interactant_field_id,
            "user_id": user_id
        }
        """
        if isinstance(data, dict):
            data=[data]

        self.qb.set_item("fields_to_fields", data)

    def rm_link_module_field(self, module_id: str, field_id: str, user_id: str):
        self.qb.rm_module_link(
            module_id=module_id,
            linked_id=field_id,
            user_id=user_id,
            table_name=self.MODULE_TO_FIELD_TABLE,
            linked_row_id_name="field_id"
        )


    def retrieve_session_fields(self, session_id: str, user_id: str) -> List[str]:
        # 1. Get session modules
        session_modules = self.qb.list_session_entries(
            session_id=session_id,
            user_id=user_id,
            table=f"sessions_to_modules",
            select="module_id"
        )
        module_ids = [m["module_id"] for m in session_modules]
        
        if not module_ids:
            return []

        # 2. Get fields for these modules
        query = f"SELECT DISTINCT field_id FROM {self.qb._table_ref(self.MODULE_TO_FIELD_TABLE)} WHERE module_id IN (SELECT unnest(?)) AND user_id = ? AND (status != 'deleted' OR status IS NULL)"
        result = self.qb.db.run_query(query, conv_to_dict=True, params=[module_ids, user_id])
        return [r["field_id"] for r in result]


    def upload_sm_fields(self, user_id: str):
        print(f"Uploading SM fields for user {user_id}")
        qfu = QFUtils()
        batch_data = []

        all_fields = FERMIONS + G_FIELDS + H

        for field in all_fields:
            try:
                # batch_field_single returns a dict of attributes (params)
                params = qfu.batch_field_single(field, dim=3)
                
                batch_data.append({
                    "id": field,
                    "params": params,
                    # user_id added in set_field
                })
            except Exception as e:
                print(f"Error preparing field {field}: {e}")

        if batch_data:
            self.set_field(batch_data, user_id)
            print(f"Uploaded {len(batch_data)} SM fields")

_default_bqcore = BQCore(dataset_id="QBRAIN")
_default_field_manager = FieldsManager(get_qbrain_table_manager(_default_bqcore))
fields_manager = _default_field_manager  # backward compat

# HANDLERS

def handle_send_users_fields(data=None, auth=None):
    """Retrieve all fields owned by a user. Required: user_id (auth or data)."""
    from core.managers_context import get_field_manager
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    if err := require_param(user_id, "user_id"):
        return err
    return {"type": "LIST_USERS_FIELDS", "data": {"fields": get_field_manager().get_fields_by_user(user_id)}}


def handle_list_users_fields(data=None, auth=None):
    """Alias for handle_send_users_fields."""
    return handle_send_users_fields(data=data, auth=auth)


def handle_send_modules_fields(data=None, auth=None):
    """Retrieve all fields linked to a module. Required: user_id, module_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    module_id = get_val(data, auth, "module_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(module_id, "module_id"):
        return err
    from core.managers_context import get_field_manager
    fields = get_field_manager().get_fields_by_module(module_id, user_id)
    return {"type": "GET_MODULES_FIELDS", "data": {"fields": fields}, "auth": {"module_id": module_id}}


def handle_get_modules_fields(data=None, auth=None):
    """Alias for handle_send_modules_fields."""
    return handle_send_modules_fields(data=data, auth=auth)


def handle_rm_link_module_field(data=None, auth=None):
    """Remove the link between a module and a field. Required: user_id, module_id, field_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    module_id = get_val(data, auth, "module_id")
    field_id = get_val(data, auth, "field_id")
    session_id = get_val(data, auth, "session_id")
    env_id = get_val(data, auth, "env_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(module_id, "module_id"):
        return err
    if err := require_param(field_id, "field_id"):
        return err
    from core.managers_context import get_field_manager
    get_field_manager().rm_link_module_field(module_id, field_id, user_id)
    fields_resp = handle_send_modules_fields(data={"module_id": module_id}, auth={"user_id": user_id})
    # Align with frontend: RM_LINK_MODULE_FIELD expects type, auth, and data
    auth_out = {"module_id": module_id, "field_id": field_id}
    if session_id:
        auth_out["session_id"] = session_id
    if env_id:
        auth_out["env_id"] = env_id
    return {
        "type": "RM_LINK_MODULE_FIELD",
        "auth": auth_out,
        "data": fields_resp.get("data", {}),
    }


def handle_get_sessions_fields(data=None, auth=None):
    """Retrieve all fields linked to a session. Required: user_id, session_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    from core.managers_context import get_field_manager
    field_ids = get_field_manager().retrieve_session_fields(session_id, user_id)
    return {"type": "SESSIONS_FIELDS", "data": {"fields": field_ids}}


def handle_list_modules_fields(data=None, auth=None):
    """Alias for handle_send_modules_fields."""
    return handle_send_modules_fields(data=data, auth=auth)


def handle_del_field(data=None, auth=None):
    """Delete a field by ID. Required: user_id, field_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    field_id = get_val(data, auth, "field_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(field_id, "field_id"):
        return err
    from core.managers_context import get_field_manager
    get_field_manager().delete_field(field_id, user_id)
    return handle_send_users_fields(data={}, auth={"user_id": user_id})


def handle_set_field(data=None, auth=None):
    """Create or update a field. Required: user_id (auth), field (data). Optional: original_id (auth)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    original_id = get_val(data, auth, "original_id")
    field_data = data.get("field") if isinstance(data, dict) else None
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param_truthy(field_data, "field"):
        return err
    from core.managers_context import get_field_manager
    fm = get_field_manager()
    if original_id:
        fm.delete_field(original_id, user_id)
    if isinstance(field_data, dict) and "linked_fields" in field_data:
        field_data = {**field_data, "interactant_fields": field_data["linked_fields"]}
        field_data.pop("linked_fields", None)
    fm.set_field(field_data, user_id)
    return handle_send_users_fields(data={}, auth={"user_id": user_id})


def handle_link_module_field(data=None, auth=None):
    """Link a field to a module. Required: user_id, module_id, field_id. Optional: session_id, env_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    module_id = get_val(data, auth, "module_id")
    field_id = get_val(data, auth, "field_id")
    session_id = get_val(data, auth, "session_id")
    env_id = get_val(data, auth, "env_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(module_id, "module_id"):
        return err
    if err := require_param(field_id, "field_id"):
        return err
    link_data = [{
        "id": generate_numeric_id(),
        "module_id": module_id,
        "field_id": field_id,
        "session_id": session_id,
        "env_id": env_id,
        "user_id": user_id,
        "status":"active"
    }]

    from core.managers_context import get_field_manager
    get_field_manager().link_module_field(link_data)

    fields_resp = handle_send_modules_fields(data={"module_id": module_id}, auth={"user_id": user_id})
    # Align with frontend: LINK_MODULE_FIELD expects type and data (sessions + fields)
    if session_id and env_id:
        return {
            "type": "LINK_MODULE_FIELD",
            "auth": {"session_id": session_id, "env_id": env_id, "module_id": module_id, "field_id": field_id},
            "data": {
                "sessions": {
                    session_id: {
                        "envs": {
                            env_id: {
                                "modules": {
                                    module_id: {
                                        "fields": [field_id]
                                    }
                                }
                            }
                        }
                    }
                },
                **fields_resp.get("data", {}),
            },
        }
    return {**fields_resp, "type": "LINK_MODULE_FIELD"}


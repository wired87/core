import base64
import random
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from google.cloud import bigquery
from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from core.fields_manager.prompt import extract_fields_prompt
from core.fields_manager.set_type import SetFieldItem

from core.qbrain_manager import QBrainTableManager
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

class FieldsManager(BQCore):
    DATASET_ID = "QBRAIN"
    FIELDS_TABLE = "fields"
    FIELDS_TO_FIELDS_TABLE = "fields_to_fields"
    MODULE_TO_FIELD_TABLE = "module_to_field"

    def __init__(self):
        super().__init__(dataset_id=self.DATASET_ID)
        self.qb = QBrainTableManager()
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
        try:
            print(f"{_FIELDS_DEBUG} _ensure_fields_table: checking")
            table_ref = f"{self.pid}.{self.DATASET_ID}.{self.FIELDS_TABLE}"
            self.bqclient.get_table(table_ref)
            print(f"{_FIELDS_DEBUG} _ensure_fields_table: exists")
        except Exception as e:
            print(f"{_FIELDS_DEBUG} _ensure_fields_table: creating: {e}")
            table = bigquery.Table(table_ref, schema=FIELD_SCHEMA)
            self.bqclient.create_table(table)
            print(f"{_FIELDS_DEBUG} _ensure_fields_table: created")

    def _ensure_fields_to_fields_table(self):
        table_ref = f"{self.pid}.{self.DATASET_ID}.{self.FIELDS_TO_FIELDS_TABLE}"
        try:
            self.bqclient.get_table(table_ref)
        except Exception as e:
            print(f"Error ensuring fields_to_fields table: {e}")
            table = bigquery.Table(table_ref, schema=FIELD_TO_FIELD_SCHEMA)
            self.bqclient.create_table(table)


    def get_field_interactants(self, field_id: str, select: str = "*") -> Dict[str, list[str or None]]:
        """
        Get environments for a session.
        """
        try:
            # todo optiize
            # 1. Receive sessions_to_envs-table rows
            query = f"""
                SELECT {select} FROM `{self.pid}.{self.DATASET_ID}.{self.FIELDS_TO_FIELDS_TABLE}`
                WHERE field_id = @field_id AND (status != 'deleted' OR status IS NULL)
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("field_id", "STRING", str(field_id)),
                ]
            )

            result = self.run_query(query, conv_to_dict=True, job_config=job_config)
            interactant_ids = [item["interactant_field_id"] for item in result]
            return {"interactant_ids": interactant_ids}
        except Exception as e:
            print(f"{_FIELDS_DEBUG} get_field_interactants: error: {e}")
            import traceback
            traceback.print_exc()
            return {"interactant_ids": []}


    def _ensure_module_to_field_table(self):
        table_ref = f"{self.pid}.{self.DATASET_ID}.{self.MODULE_TO_FIELD_TABLE}"
        try:
            self.bqclient.get_table(table_ref)
        except Exception as e:
            print(f"Error ensuring module_to_field table: {e}")
            table = bigquery.Table(table_ref, schema=MODULE_TO_FIELD_SCHEMA)
            self.bqclient.create_table(table)


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
                nid=field_ids,
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
                nid=field_id,
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
            nid=field_ids,
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
        """
        "id": "STRING",
        "user_id": "STRING",
        "status": "STRING",
        "params": "STRING",
        "created_at": "TIMESTAMP",
        "updated_at": "TIMESTAMP",
        """
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
        query = f"""
            SELECT DISTINCT field_id FROM `{self.pid}.{self.DATASET_ID}.{self.MODULE_TO_FIELD_TABLE}`
            WHERE module_id IN UNNEST(@module_ids) AND user_id = @user_id AND (status != 'deleted' OR status IS NULL)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("module_ids", "STRING", module_ids),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        result = self.run_query(query, conv_to_dict=True, job_config=job_config)
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

fields_manager = FieldsManager()

# HANDLERS

def handle_send_users_fields(payload):
    # SEND_USERS_FIELDS: get fields-table-rows ... -> type="list_field_user"
    print("handle_send_users_fields...")
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    
    if not user_id:
        return {"error": "Missing user_id"}
    
    fields = fields_manager.get_fields_by_user(user_id)
    #print("fields", fields)
    return {
        "type": "LIST_USERS_FIELDS",
        "data": {"fields": fields}
    }

def handle_list_users_fields(payload):
    print("handle_list_users_fields...")
    return handle_send_users_fields(payload)

def handle_send_modules_fields(payload):
    # SEND_MODULES_FIELDS
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    module_id = auth.get("module_id")
    if not user_id or not module_id:
        return {"error": "Missing user_id or module_id"}
    
    fields = fields_manager.get_fields_by_module(module_id, user_id)
    return {
        "type": "GET_MODULES_FIELDS", 
        "data": {"fields": fields},
        "auth": {"module_id": module_id}
    }

def handle_get_modules_fields(payload):
    return handle_send_modules_fields(payload)

def handle_rm_link_module_field(payload):
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    module_id = auth.get("module_id")
    field_id = auth.get("field_id")
    
    if not all([user_id, module_id, field_id]):
        return {"error": "Missing params"}
        
    fields_manager.rm_link_module_field(module_id, field_id, user_id)
    return handle_send_modules_fields(payload)

def handle_get_sessions_fields(payload):
    # receive type="", auth={user_id:str, session_id:str} -> get session fields
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    
    if not user_id or not session_id:
        return {"error": "Missing params"}
        
    field_ids = fields_manager.retrieve_session_fields(session_id, user_id)
    return {
        "type": "SESSIONS_FIELDS",
        "data": {"fields": field_ids}
    }

def handle_list_modules_fields(payload):
    # receive "LIST_MODULES_FIELDS" -> send_fields_module
    return handle_send_modules_fields(payload) # Assuming send_fields_module maps to this logic

def handle_del_field(payload):
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    field_id = auth.get("field_id")
    if not user_id or not field_id:
        return {"error": "Missing user_id or field_id"}
    
    fields_manager.delete_field(field_id, user_id)
    return handle_send_users_fields(payload)

def handle_set_field(payload):
    auth = payload.get("auth", {})
    data = payload.get("data", {})
    user_id = auth.get("user_id")
    
    field_data = data.get("field")
    
    if not user_id or not field_data:
        return {"error": "Missing user_id or field data"}
    
    original_id = auth.get("original_id")
    if original_id:
        fields_manager.delete_field(original_id, user_id)

    if "linked_fields" in field_data:
        field_data["interactant_fields"] = field_data["linked_fields"]
        field_data.pop("linked_fields")

    fields_manager.set_field(field_data, user_id)
    return handle_send_users_fields(payload)

def handle_link_module_field(payload):
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    module_id = auth.get("module_id")
    field_id = auth.get("field_id")
    session_id = auth.get("session_id")
    env_id = auth.get("env_id")
    
    if not all([user_id, module_id, field_id]):
        return {"error": "Missing params"}
        
    link_data = [{
        "id": generate_numeric_id(),
        "module_id": module_id,
        "field_id": field_id,
        "session_id": session_id,
        "env_id": env_id,
        "user_id": user_id,
        "status":"active"
    }]

    fields_manager.link_module_field(link_data)
    
    # Return delta structure if session info is present
    if session_id and env_id:
        return {
            "type": "ENABLE_SM", # Using same type as SM enable for consistent frontend handling? Or unique? User asked for same format.
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
                }
            }
        }

    return handle_send_modules_fields(payload)


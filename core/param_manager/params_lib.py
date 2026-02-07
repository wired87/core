import base64
import random
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from google.cloud import bigquery
from a_b_c.bq_agent._bq_core.bq_handler import BQCore

from core.qbrain_manager import QBrainTableManager
from qf_utils.all_subs import FERMIONS, G_FIELDS, H
from qf_utils.qf_utils import QFUtils

_PARAMS_DEBUG = "[ParamsManager]"


def generate_numeric_id() -> str:
    """Generate a random numeric ID."""
    return str(random.randint(1000000000, 9999999999))

class ParamsManager(BQCore):
    DATASET_ID = "QBRAIN"
    PARAMS_TABLE = "params"
    FIELDS_TO_PARAMS_TABLE = "fields_to_params"

    def __init__(self):
        super().__init__(dataset_id=self.DATASET_ID)
        self.qb = QBrainTableManager()
        self.table = f"{self.PARAMS_TABLE}"
        from core.param_manager import case as param_case
        set_case = next((c for c in param_case.RELAY_PARAM if c.get("case") == "SET_PARAM"), None)
        req_struct = set_case.get("req_struct", {}) if set_case else {}
        out_struct = set_case.get("out_struct", {}) if set_case else {}
        self._extract_prompt = f"""Extract parameter definitions from the provided file content.

Input structure (what you receive): raw file bytes decoded as text or document content.
Output structure (return valid JSON only):
  req_struct: {json.dumps(req_struct, indent=2)}
  out_struct: {json.dumps(out_struct, indent=2)}

Return a JSON object with a "param" key (or "params" list) matching the data.param shape expected by SET_PARAM.
Each param dict should have: id, name, param_type (e.g. FLOAT64, STRING), description, optionally const/is_constant, value.
Output valid JSON only, no markdown."""

    def get_axis(self, params:dict):
        # get axis for pasm from BQ
        return [self.get_axis_param(p["const"]) for p in params]

    def get_axis_param(self, const:bool):
        return None if const is True else 1

    def extract_from_file_bytes(self, file_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Extract manager-specific param content from file bytes using the static prompt.
        Uses Gem LLM with req_struct/out_struct from SET_PARAM case.
        """
        try:
            from gem_core.gem import Gem
            gem = Gem()
            try:
                text_content = file_bytes.decode("utf-8")
                content = f"{self._extract_prompt}\n\n--- FILE CONTENT ---\n{text_content}"
                response = gem.ask(content)
            except UnicodeDecodeError:
                b64 = base64.b64encode(file_bytes).decode("ascii")
                response = gem.ask_mm(file_content_str=b64, prompt=self._extract_prompt)
            text = (response or "").strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            if "param" in parsed:
                return {"param": parsed["param"]}
            if "params" in parsed:
                return {"param": parsed["params"] if isinstance(parsed["params"], list) else parsed["params"]}
            return {"param": parsed}
        except Exception as e:
            print(f"{_PARAMS_DEBUG} extract_from_file_bytes error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_users_params(self, user_id: str, select: str = "*") -> List[Dict[str, Any]]:
        print("get_users_params", user_id)
        result = self.qb.get_users_entries(
            user_id=user_id,
            table=self.table,
            select=select
        )
        return [dict(row) for row in result]

    def set_param(
            self,
            param_data: Dict[str, Any] or List[dict],
            user_id: str,
    ):
        """
        Upsert parameters.
        Schema: id, name, type, user_id, description, embedding (ARRAY<FLOAT64>), status, created_at, updated_at
        """
        try:
            print(f"{_PARAMS_DEBUG} set_param: user_id={user_id}, count={1 if not isinstance(param_data, list) else len(param_data)}")
            if not isinstance(param_data, list):
                param_data = [param_data]
            prev_params = []
            for p in param_data:
                p["user_id"] = user_id
                if "is_constant" in p:
                    p["const"] = p["is_constant"]
                if "value" in p:
                    p["value"] = json.dumps(p["value"])
                if "axis_def" not in p or p.get("axis_def") is None and "const" in p:
                    p["axis_def"] = self.get_axis_param(p["const"])
                if "embedding" in p and p["embedding"]:
                    if isinstance(p["embedding"], str):
                        try:
                            p["embedding"] = json.loads(p["embedding"])
                        except Exception:
                            pass
                prev_param = p.copy()
                prev_param["id"] = f"prev_{p['id']}"
                prev_param["description"] = "Prev variation to trac emergence over time. The val field is empty and taks the prev val of its parent at each iteration"
                prev_param["value"] = None
                prev_params.append(prev_param)
            self.qb.set_item(self.PARAMS_TABLE, param_data)
            print(f"{_PARAMS_DEBUG} set_param: done")
        except Exception as e:
            print(f"{_PARAMS_DEBUG} set_param: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def delete_param(self, param_id: str, user_id: str):
        try:
            print(f"{_PARAMS_DEBUG} delete_param: param_id={param_id}, user_id={user_id}")
            self.qb.del_entry(
                nid=param_id,
                table=self.table,
                user_id=user_id
            )
            print(f"{_PARAMS_DEBUG} delete_param: done")
        except Exception as e:
            print(f"{_PARAMS_DEBUG} delete_param: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def link_field_param(self, data: List[Dict[str, Any]] or Dict[str, Any], user_id: str):
        """
        Link field to param.
        Schema: id, field_id, param_id, param_value (STRING), user_id, status, ...
        """
        if isinstance(data, dict):
            data = [data]

        now = datetime.now().isoformat()
        
        for item in data:
            row = {
                "id": item.get("id", generate_numeric_id()),
                "field_id": item["field_id"],
                "param_id": item["param_id"],
                "param_value": str(item.get("param_value", "")), # Store value as string
                "user_id": user_id,
                "status": "active",
                "created_at": now,
                "updated_at": now
            }
            self.qb.set_item(self.FIELDS_TO_PARAMS_TABLE, row, keys={"id": row["id"]})

    def rm_link_field_param(self, field_id: str, param_id: str, user_id: str):
        """
        Remove link between field and param.
        Using upsert_copy with custom matching since we might want to target specfic link?
        But table has 'id'. If we don't have link ID, we match by field_id + param_id.
        Schema of fields_to_params has 'id'.
        
        If we want to delete by field_id + param_id:
        """
        
        # We need a custom query or strict logic. QBrainTableManager.rm_module_link uses upsert_copy with keys.
        # Let's use similar logic. 
        # But we have two keys: field_id AND param_id.
        
        keys = {
            "field_id": field_id,
            "param_id": param_id,
            "user_id": user_id
        }
        updates = {"status": "deleted"}
        self.qb.set_item(self.FIELDS_TO_PARAMS_TABLE, updates, keys=keys)

    def get_fields_params(self, field_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Get info about params linked to a field.
        We probably want the param details AND the value from the link table.
        """
        
        # 1. Get links
        query = f"""
            SELECT * FROM `{self.pid}.{self.DATASET_ID}.{self.FIELDS_TO_PARAMS_TABLE}`
            WHERE field_id = @field_id AND user_id = @user_id AND (status != 'deleted' OR status IS NULL)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("field_id", "STRING", field_id),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        links = self.run_query(query, conv_to_dict=True, job_config=job_config)
        
        if not links:
            return []

        # 2. Get param details
        param_ids = [l["param_id"] for l in links]
        # De-duplicate IDs
        param_ids = list(set(param_ids))
        
        params_details = self.qb.row_from_id(
            nid=param_ids,
            table=self.PARAMS_TABLE,
            select="*"
        )
        
        # 3. Merge details
        # We want to return a list of params, perhaps with 'value' attached?
        # Or return links + expanded param info?
        
        params_map = {p["id"]: p for p in params_details}
        
        result = []
        for l in links:
            pid = l["param_id"]
            if pid in params_map:
                p_info = params_map[pid].copy()
                # Attach value from link
                p_info["link_value"] = l.get("param_value")
                p_info["link_id"] = l.get("id")
                result.append(p_info)
                
        return result


    def upload_sm_params(self, user_id: str):
        print(f"Uploading SM params for user {user_id}")
        qfu = QFUtils()
        collected_params = {}

        all_fields = FERMIONS + G_FIELDS + H

        for field in all_fields:
            try:
                # batch_field_single returns a dict of attributes (params)
                params = qfu.batch_field_single(field, dim=3)
                
                if isinstance(params, dict):
                    for k, v in params.items():
                        # Determine BQ type
                        bq_type = "STRING"
                        if isinstance(v, (int, float)):
                            bq_type = "FLOAT64"
                        elif isinstance(v, bool):
                            bq_type = "BOOL"
                            
                        if k not in collected_params:
                            collected_params[k] = bq_type
            except Exception as e:
                print(f"Error extracting params for {field}: {e}")

        # Batch upsert
        batch_data = []
        for p_name, p_type in collected_params.items():
            batch_data.append({
                "id": p_name, # Use name as ID for standard params
                "name": p_name,
                "param_type": p_type,
                "description": f"Standard Model Parameter: {p_name}"
            })

        if batch_data:
            self.set_param(batch_data, user_id)
            print(f"Uploaded {len(batch_data)} SM params")

params_manager = ParamsManager()

def handle_get_users_params(payload):
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    if not user_id:
        return {"error": "Missing user_id"}
    
    params = params_manager.get_users_params(user_id)
    return {
        "type": "LIST_USERS_PARAMS",
        "data": {"params": params}
    }

def handle_list_users_params(payload):
    return handle_get_users_params(payload)
"""
each p gets prevp (pp)
"""
def handle_set_param(payload):
    auth = payload.get("auth", {})
    data = payload.get("data", {})
    user_id = auth.get("user_id")
    param_data = data.get("param")
    
    if not user_id or not param_data:
        return {"error": "Missing user_id or param data"}
        
    original_id = auth.get("original_id")
    if original_id:
        params_manager.delete_param(original_id, user_id)

    params_manager.set_param(param_data, user_id)
    return handle_get_users_params(payload) # Return updated list

def handle_del_param(payload):
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    param_id = auth.get("param_id")
    
    if not user_id or not param_id:
        return {"error": "Missing params"}
        
    params_manager.delete_param(param_id, user_id)
    return handle_get_users_params(payload)


def handle_link_field_param(payload):
    auth = payload.get("auth", {})
    data = payload.get("data", {})
    user_id = auth.get("user_id")
    
    # data can be {field_id, param_id, value} or list of such
    links = data.get("links")
    if not links:
        # fallback if passed flat
        if "field_id" in auth and "param_id" in auth:
             links = [{
                 "field_id": auth["field_id"],
                 "param_id": auth["param_id"],
                 "param_value": data.get("param_value")
             }]
        elif "field_id" in data and "param_id" in data:
             links = [data]
    
    if not user_id or not links:
        return {"error": "Missing data"}

    params_manager.link_field_param(links, user_id)
    
    # Return updated field params? or just params list?
    # Usually we want to see the params for the context we are in.
    # If we linked to a field, maybe we want that field's params.
    # But usually generic return or specific return.
    
    # Let's assume we return list of params for the first field involved?
    target_field_id = links[0]["field_id"]
    return {
        "type": "GET_FIELDS_PARAMS",
        "data": {"params": params_manager.get_fields_params(target_field_id, user_id)},
        "auth": {"field_id": target_field_id}
    }

def handle_rm_link_field_param(payload):
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    field_id = auth.get("field_id")
    param_id = auth.get("param_id")
    
    if not all([user_id, field_id, param_id]):
        return {"error": "Missing identifiers"}
        
    params_manager.rm_link_field_param(field_id, param_id, user_id)
    
    return {
        "type": "GET_FIELDS_PARAMS",
        "data": {"params": params_manager.get_fields_params(field_id, user_id)},
        "auth": {"field_id": field_id}
    }

def handle_get_fields_params(payload):
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    field_id = auth.get("field_id")
    
    if not user_id or not field_id:
        return {"error": "Missing field_id or user_id"}
        
    params = params_manager.get_fields_params(field_id, user_id)
    return {
        "type": "GET_FIELDS_PARAMS",
        "data": {"params": params},
        "auth": {"field_id": field_id}
    }

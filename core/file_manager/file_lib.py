
import base64
import json
import random
from datetime import datetime
from typing import Dict, Any, List, TypedDict

import dotenv
from core.file_manager.extractor import RawModuleExtractor
from core.param_manager.params_lib import ParamsManager
from core.fields_manager.fields_lib import FieldsManager
from core.method_manager.method_lib import MethodManager

from core.qbrain_manager import QBrainTableManager
from a_b_c.bq_agent._bq_core.bq_handler import BQCore
dotenv.load_dotenv()


class EquationContent(TypedDict, total=False):
    """Struct for extracted math/LaTeX from PDF preprocessing."""
    latex: List[str]
    math_elements: List[str]
    equations: List[str]


def _files_to_bytes(files: List) -> List[bytes]:
    """Convert file payloads (base64 strings, bytes, or file-like) to bytes list."""
    result = []
    for f in files or []:
        try:
            if isinstance(f, bytes):
                result.append(f)
            elif isinstance(f, str):
                s = f.split("base64,")[-1] if "base64," in f else f
                result.append(base64.b64decode(s))
            elif hasattr(f, "read"):
                result.append(f.read())
            else:
                continue
        except Exception as e:
            print(f"[FileManager] files_to_bytes skip item: {e}")
    print(f"[FileManager] files_to_bytes: converted {len(result)} file(s) to bytes")
    return result


class FileManager(BQCore, RawModuleExtractor):
    DATASET_ID = "QBRAIN"
    MODULES_TABLE = "modules"
    FILES_TABLE = "files"

    def __init__(self):
        super().__init__(dataset_id=self.DATASET_ID)
        RawModuleExtractor.__init__(self)
        self.qb = QBrainTableManager()
        self.param_manager = ParamsManager()
        self.fields_manager = FieldsManager()
        self.method_manager = MethodManager()

    file_params: dict = {}  # Testing only attribute



    def _req_struct_to_json_schema(self, req_struct: dict, content_type: str, data_key: str) -> dict:
        """
        Convert req_struct from SET case to JSON Schema for Gemini.
        Uses the exact req_struct as the Tool schema for the output.
        """
        data_struct = req_struct.get("data", req_struct) or {}
        item_def = data_struct.get(data_key, data_struct) if isinstance(data_struct, dict) else data_struct

        def dict_to_schema(d: dict) -> dict:
            props = {}
            for k, v in d.items():
                desc = str(v) if isinstance(v, str) else ""
                if isinstance(v, str) and "list" in v:
                    props[k] = {"type": "array", "items": {"type": "string"}, "description": desc}
                elif isinstance(v, str) and "dict" in v:
                    props[k] = {"type": "object", "additionalProperties": True, "description": desc}
                elif isinstance(v, dict):
                    props[k] = dict_to_schema(v)
                else:
                    props[k] = {"type": "string", "description": desc}
            return {"type": "object", "properties": props, "additionalProperties": True}

        if isinstance(item_def, dict) and item_def:
            item_schema = dict_to_schema(item_def)
            item_schema["description"] = f"Single item for SET_{content_type.upper()} handler"
            req_keys = {"param": ["name"], "field": ["name"], "method": ["equation"]}
            if content_type in req_keys:
                item_schema["required"] = [r for r in req_keys[content_type] if r in (item_schema.get("properties") or {})]
        else:
            # req_struct has string type hint (e.g. param: "dict|list"); use explicit schema from case
            item_schema = {
                "type": "object",
                "description": f"Single item for SET_{content_type.upper()} handler",
                "additionalProperties": True,
            }

        return {
            "type": "object",
            "description": f"Output matching SET_{content_type.upper()} req_struct",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Extracted items for handler",
                    "items": item_schema,
                },
            },
            "required": ["items"],
        }

    def _extract_with_struct(
        self,
        equation_content: EquationContent,
        user_prompt: str,
        content_type: str,
        pdf_bytes: bytes = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract items using the exact req_struct of the SET method as Tool (JSON schema) for Gemini.
        Inputs: file bytes (PDF) + user prompt. Sys instructions guide generation for the specific handler.
        """
        print(f"[FileManager] _extract_with_struct: starting content_type={content_type}")
        from gem_core.gem import Gem
        case_map = {
            "param": ("core.param_manager.case", "RELAY_PARAM", "SET_PARAM", "param"),
            "field": ("core.fields_manager.case", "RELAY_FIELD", "SET_FIELD", "field"),
            "method": ("core.method_manager.case", "RELAY_METHOD", "SET_METHOD", "data"),
        }
        mod_name, attr, case_name, data_key = case_map[content_type]
        mod = __import__(mod_name, fromlist=[attr])
        relay = getattr(mod, attr, [])
        set_case = next((c for c in relay if c.get("case") == case_name), None)
        req_struct = set_case.get("req_struct", {}) if set_case else {}
        json_schema = self._req_struct_to_json_schema(req_struct, content_type, data_key)
        print(f"[FileManager] _extract_with_struct: loaded req_struct for {content_type}, schema keys: {list(json_schema.get('properties', {}).keys())}")

        # System instructions: generate data for the specific SET method
        set_name = f"SET_{content_type.upper()}"
        sys_instructions = f"""You are extracting structured data from a scientific document for the {set_name} handler.
Your output must conform exactly to the provided JSON schema (Tool), which matches the handler's req_struct.

User instructions: {user_prompt or 'Extract all relevant content of this type.'}

Pre-extracted equation content from the document (use as context; the PDF is also attached):
- latex: {json.dumps(equation_content.get('latex', [])[:50], default=str)}
- math_elements: {json.dumps(equation_content.get('math_elements', [])[:50], default=str)}
- equations: {json.dumps(equation_content.get('equations', [])[:20], default=str)}

Generate a JSON object with an "items" array. Each item must match the schema exactly.
Return only valid JSON, no markdown or extra text."""

        config = {
            "response_mime_type": "application/json",
            "response_json_schema": json_schema,
        }

        try:
            gem = Gem()
            if pdf_bytes:
                try:
                    print(f"[FileManager] _extract_with_struct: calling Gemini (multimodal: PDF + schema Tool)")
                    b64 = base64.b64encode(pdf_bytes).decode("ascii")
                    response = gem.ask_mm(
                        file_content_str=f"data:application/pdf;base64,{b64}",
                        prompt=sys_instructions,
                        config=config,
                    )
                except Exception as e1:
                    print(f"[FileManager] _extract_with_struct: multimodal failed ({e1}), falling back to text-only")
                    response = gem.ask(sys_instructions, config=config)
            else:
                print(f"[FileManager] _extract_with_struct: calling Gemini (text-only + schema Tool)")
                response = gem.ask(sys_instructions, config=config)
            text = (response or "").strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            items = parsed.get("items", parsed) if isinstance(parsed, dict) else parsed
            result = items if isinstance(items, list) else [items]
            print(f"[FileManager] _extract_with_struct: {content_type} done -> {len(result)} item(s)")
            return result
        except Exception as e:
            print(f"[FileManager] _extract_with_struct {content_type} error: {e}")
            return []

    def process_and_upload_file_config(
        self,
        user_id: str,
        data: Dict[str, Any],
        testing: bool = False,
        mock_extraction: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract case-specific content from files via param/field/method managers,
        upsert via their set methods, upsert file metadata to files table.
        Returns type=CONTENT_EXTRACTED, data={content_type: list[ids]}, created_components.

        mock_extraction: If True, skip Gemini and use sample data (for quick tests).
        """
        print("[FileManager] process_and_upload_file_config: starting")
        module_id = data.get("id") or f"file_{random.randint(100000, 999999)}"
        files_raw = data.get("files", [])
        file_bytes_list = _files_to_bytes(files_raw)
        user_prompt = data.get("prompt", "") or data.get("msg", "")
        print(f"[FileManager] process_and_upload_file_config: module_id={module_id}, files={len(file_bytes_list)}, user_prompt={bool(user_prompt)}, mock_extraction={mock_extraction}")

        # 0. Preprocessing: extract math/LaTeX from PDFs into equation_content
        files_b64 = [
            base64.b64encode(b).decode("ascii")
            for b in file_bytes_list
        ]
        equation_content = "\n".join(files_b64)
        print("[FileManager] process_and_upload_file_config: preprocessing done")

        extracted_ids: Dict[str, List[str]] = {"param": [], "field": [], "method": []}
        fallback_users_params = self.param_manager.get_users_params(user_id)

        # --- PARAM EXTRACTION ---
        params_raw = self.param_manager.extract_from_file_bytes(
            content=equation_content,
            instructions=user_prompt,
            users_params=fallback_users_params,
        )
        params_list: List[Dict[str, Any]] = []
        if isinstance(params_raw, dict):
            _p = params_raw.get("param") or params_raw.get("params")
            if isinstance(_p, list):
                params_list = [p for p in _p if isinstance(p, dict)]
            elif isinstance(_p, dict):
                params_list = [_p]
        elif isinstance(params_raw, list):
            params_list = [p for p in params_raw if isinstance(p, dict)]

        ### METHOD EXTRACTION
        methods_raw = self.method_manager.extract_from_file_bytes(
            content=equation_content,
            instructions=user_prompt,
            params=params_list,
            fallback_params=fallback_users_params,
        )
        methods_list: List[Dict[str, Any]] = []
        if isinstance(methods_raw, dict):
            _m = methods_raw.get("methods")
            if isinstance(_m, list):
                methods_list = [m for m in _m if isinstance(m, dict)]
            elif isinstance(_m, dict):
                methods_list = [_m]
        elif isinstance(methods_raw, list):
            methods_list = [m for m in methods_raw if isinstance(m, dict)]

        # --- FIELD EXTRACTION ---
        fields_raw = self.fields_manager.extract_from_file_bytes(
            equation_content,
            params_list,
            user_prompt,
            fallback_users_params,
        )
        fields_list: List[Dict[str, Any]] = []
        if isinstance(fields_raw, dict):
            _f = fields_raw.get("field") or fields_raw.get("fields")
            if isinstance(_f, list):
                fields_list = [f for f in _f if isinstance(f, dict)]
            elif isinstance(_f, dict):
                fields_list = [_f]
        elif isinstance(fields_raw, list):
            fields_list = [f for f in fields_raw if isinstance(f, dict)]


        # Collect created components in exact handler input format
        created_components: Dict[str, Any] = {"param": [], "field": [], "method": []}

        # 2. Params
        if params_list:
            ids = []
            for item in params_list:
                pid = item.get("id") or item.get("name") or str(random.randint(100000, 999999))
                item["id"] = pid
                item["name"] = item.get("name") or pid
                ids.append(pid)
            if not testing:
                self.param_manager.set_param(params_list, user_id)
            extracted_ids["param"].extend(ids)
            created_components["param"] = [{"auth": {"user_id": user_id}, "data": {"param": params_list}}]
            print(f"[FileManager] process_and_upload_file_config: params upserted -> {ids}")

        # 3. Fields
        if fields_list:
            ids = []
            for item in fields_list:
                fid = item.get("id") or str(random.randint(100000, 999999))
                item["id"] = fid
                ids.append(fid)
            if not testing:
                self.fields_manager.set_field(fields_list, user_id)
            extracted_ids["field"].extend(ids)
            created_components["field"] = [
                {"auth": {"user_id": user_id}, "data": {"field": f}} for f in fields_list
            ]
            print(f"[FileManager] process_and_upload_file_config: fields upserted -> {ids}")

        # 4. Methods
        if methods_list:
            ids = []
            for m in methods_list:
                mid = m.get("id") or str(random.randint(100000, 999999))
                m["id"] = mid
                m["user_id"] = user_id
                if "equation" in m and "code" not in m:
                    m["code"] = m["equation"]
                ids.append(mid)
            if not testing:
                self.method_manager.set_method(methods_list, user_id)
            extracted_ids["method"].extend(ids)
            created_components["method"] = [{"auth": {"user_id": user_id}, "data": dict(m)} for m in methods_list]
            print(f"[FileManager] process_and_upload_file_config: methods upserted -> {ids}")

        # 5. Upsert file record(s) to files table
        if testing or not file_bytes_list:
            print("[FileManager] process_and_upload_file_config: skipping files table (testing or no files)")
        if not testing and file_bytes_list:
            for i, _ in enumerate(file_bytes_list):
                file_row = {
                    "id": f"{module_id}_f{i}" if len(file_bytes_list) > 1 else module_id,
                    "user_id": user_id,
                    "module_id": module_id,
                    "created_at": datetime.utcnow().isoformat(),
                }
                try:
                    self.qb.set_item(self.FILES_TABLE, file_row)
                    print(f"[FileManager] process_and_upload_file_config: file row upserted -> {file_row.get('id')}")
                except Exception as e:
                    print(f"[FileManager] files table upsert skipped: {e}")

        # 6. Upsert module
        if not testing:
            row = {**data, "user_id": user_id}
            row.pop("methods", None)
            row.pop("fields", None)
            row.pop("files", None)  # files are not JSON-serializable (handles/bytes)
            self.set_module(row, user_id)
            print("[FileManager] process_and_upload_file_config: module upserted")

        print(f"[FileManager] process_and_upload_file_config: done -> param={len(extracted_ids['param'])}, field={len(extracted_ids['field'])}, method={len(extracted_ids['method'])}")
        return {
            "type": "CONTENT_EXTRACTED",
            "data": {
                "param": list(dict.fromkeys(extracted_ids["param"])),
                "field": list(dict.fromkeys(extracted_ids["field"])),
                "method": list(dict.fromkeys(extracted_ids["method"])),
            },
            "created_components": created_components,
        }

    def set_module(self, rows: List[Dict] or Dict, user_id: str):
        """
        Upsert module entry to BigQuery.
        Direct copy/adapt from ModuleWsManager.
        """
        print("[FileManager] set_module: starting")
        if isinstance(rows, dict):
            rows = [rows]

        for row in rows:
            row["user_id"] = user_id
            
            if row.get("parent"):
                row.pop("parent")

            if row.get("module_index"):
                row.pop("module_index")

        # Serializing specific fields is handled in QBrainTableManager.set_item but 
        # let's duplicate the safety check if needed or rely on qb.
        self.qb.set_item(self.MODULES_TABLE, rows)
        print(f"[FileManager] set_module: done -> {len(rows)} row(s)")

 

# Global Instance
file_manager = FileManager()

# -- VALIDATION HANDLERS --

def handle_set_file(payload):
    """
    Handle SET_FILE request.
    Extract params/code from files and upsert module.
    """
    auth = payload.get("auth", {})
    data = payload.get("data", {})
    user_id = auth.get("user_id")

    if not user_id:
        return {"error": "Missing user_id"}

    # Handle original_id deletion if provided (Module Replacement)
    original_id = auth.get("original_id")
    if original_id:
        try:
            file_manager.qb.del_entry(original_id, "modules", user_id)
        except Exception:
            pass

    result = file_manager.process_and_upload_file_config(user_id, data)
    return result

if __name__ =="__main__":
    file_manager.process_and_upload_file_config(
        user_id="public",
        data={
            "id": "hi",
            "files": [
                open(r"C:\Users\bestb\PycharmProjects\BestBrain\test_paper.pdf", "rb")
            ]
        },
        testing=False
    )
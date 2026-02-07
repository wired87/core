
import base64
import random
from datetime import datetime
from typing import Dict, Any, List

import dotenv
from core.file_manager.extractor import RawModuleExtractor
from core.param_manager.params_lib import ParamsManager
from core.fields_manager.fields_lib import FieldsManager
from core.qbrain_manager import QBrainTableManager

from a_b_c.bq_agent._bq_core.bq_handler import BQCore

dotenv.load_dotenv()


def _files_to_bytes(files: List) -> List[bytes]:
    """Convert file payloads (base64 strings or bytes) to bytes list."""
    result = []
    for f in files or []:
        try:
            if isinstance(f, bytes):
                result.append(f)
            elif isinstance(f, str):
                s = f.split("base64,")[-1] if "base64," in f else f
                result.append(base64.b64decode(s))
            else:
                continue
        except Exception as e:
            print(f"[FileManager] files_to_bytes skip item: {e}")
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
        from core.method_manager.method_lib import MethodManager
        self.method_manager = MethodManager()

    file_params: dict = {}  # Testing only attribute

    def process_and_upload_file_config(self, user_id: str, data: Dict[str, Any], testing: bool = False) -> Dict[str, Any]:
        """
        Extract case-specific content from files via param/field/method managers,
        upsert via their set methods, upsert file metadata to files table.
        Returns type=CONTENT_EXTRACTED, data={content_type: list[ids]}.
        """
        module_id = data.get("id") or f"file_{random.randint(100000, 999999)}"
        files_raw = data.get("files", [])
        file_bytes_list = _files_to_bytes(files_raw)

        extracted_ids: Dict[str, List[str]] = {"param": [], "field": [], "method": []}

        for file_bytes in file_bytes_list:
            # 1. Extract via each manager
            param_out = self.param_manager.extract_from_file_bytes(file_bytes)
            field_out = self.fields_manager.extract_from_file_bytes(file_bytes)
            method_out = self.method_manager.extract_from_file_bytes(file_bytes)

            # 2. Params
            if param_out and param_out.get("param"):
                p = param_out["param"]
                params_list = p if isinstance(p, list) else [p]
                ids = []
                for item in params_list:
                    pid = item.get("id") or item.get("name") or str(random.randint(100000, 999999))
                    item["id"] = pid
                    item["name"] = item.get("name") or pid
                    ids.append(pid)
                if not testing:
                    self.param_manager.set_param(params_list, user_id)
                extracted_ids["param"].extend(ids)

            # 3. Fields
            if field_out and field_out.get("field"):
                f = field_out["field"]
                fields_list = f if isinstance(f, list) else [f]
                ids = []
                for item in fields_list:
                    fid = item.get("id") or str(random.randint(100000, 999999))
                    item["id"] = fid
                    ids.append(fid)
                if not testing:
                    self.fields_manager.set_field(fields_list, user_id)
                extracted_ids["field"].extend(ids)

            # 4. Methods
            if method_out and method_out.get("data"):
                methods_data = method_out["data"]
                methods_list = methods_data if isinstance(methods_data, list) else [methods_data]
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

        # 5. Upsert file record(s) to files table
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
                except Exception as e:
                    print(f"[FileManager] files table upsert skipped: {e}")

        # 6. Upsert module
        if not testing:
            row = {**data, "user_id": user_id}
            row.pop("methods", None)
            row.pop("fields", None)
            self.set_module(row, user_id)

        return {
            "type": "CONTENT_EXTRACTED",
            "data": {
                "param": list(dict.fromkeys(extracted_ids["param"])),
                "field": list(dict.fromkeys(extracted_ids["field"])),
                "method": list(dict.fromkeys(extracted_ids["method"])),
            },
        }

    def set_module(self, rows: List[Dict] or Dict, user_id: str):
        """
        Upsert module entry to BigQuery.
        Direct copy/adapt from ModuleWsManager.
        """
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


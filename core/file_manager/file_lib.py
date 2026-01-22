
import dotenv
from typing import Dict, Any, List

from core.file_manager.extractor import RawModuleExtractor
from core.param_manager.params_lib import ParamsManager
from core.fields_manager.fields_lib import FieldsManager

from a_b_c.bq_agent._bq_core.bq_handler import BQCore

dotenv.load_dotenv()



class FileManager(BQCore, RawModuleExtractor):
    DATASET_ID = "QBRAIN"
    MODULES_TABLE = "modules"

    def __init__(self):
        super().__init__(dataset_id=self.DATASET_ID)
        RawModuleExtractor.__init__(self)
        self.param_manager = ParamsManager()
        self.fields_manager = FieldsManager()


    file_params: dict = {} # Testing only attribute

    def process_and_upload_file_config(self, user_id: str, data: Dict[str, Any], testing: bool = False):
        """
        Main workflow: Process files -> Extract -> Upsert Module Config.
        """
        module_id = data.get("id")
        files = data.get("files", [])
        
        # 1. Process Files & Extract Data (Inherited from RawModuleExtractor)
        extracted_data = self.process(module_id, files)
        
        # 2. Upsert Entities via Managers
        
        # Params
        params_dict = extracted_data.get("params", {})
        if params_dict:
            params_list = []
            for p_name, p_type in params_dict.items():
                params_list.append({
                    "id": p_name,
                    "name": p_name,
                    "param_type": p_type,
                    "description": f"Extracted from {module_id}",
                    "user_id": user_id
                })
            if not testing:
                self.param_manager.set_param(params_list, user_id)

        # Methods
        methods_list = extracted_data.get("methods", [])
        if methods_list:
            for m in methods_list:
                m["user_id"] = user_id
                if "nid" in m:
                    m["id"] = m["nid"]
            if not testing:
                self.method_manager.set_method(methods_list, user_id)

        # Fields (ClassVars)
        fields_list = extracted_data.get("fields", [])
        if fields_list:
            for f in fields_list:
                f["user_id"] = user_id
                if "nid" in f:
                    f["id"] = f["nid"]
            if not testing:
                self.fields_manager.set_field(fields_list, user_id)

        # 3. Upsert Module
        row = {
            **data,
            **extracted_data,
            "user_id": user_id
        }
        
        # Cleanup extracted lists from module row to avoid schema issues
        row.pop("methods", None)
        row.pop("fields", None)

        if not testing:
            print("Upserting file/module data:", module_id)
            self.set_module(row, user_id)
        else:
            print("Testing mode: Skipping upserts for module:", module_id)
        
        # Return classified data
        return {
            "module": row,
            "methods": methods_list,
            "fields": fields_list,
            "params": params_dict 
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

    # Process and Upsert
    classified_data = file_manager.process_and_upload_file_config(user_id, data)
    
    # Handle original_id deletion if provided (Module Replacement)
    original_id = auth.get("original_id")
    if original_id:
        try:
             file_manager.qb.del_entry(original_id, "modules", user_id)
        except:
             pass

    return {
        "type": "UPSERTED_MODULE_CONTENT",
        "data": classified_data
    }



import logging
import base64
import random
from datetime import datetime
from typing import Dict, Any, List, Optional

from google.cloud import bigquery
from a_b_c.bq_agent._bq_core.bq_handler import BQCore

from core.fields_manager.fields_lib import fields_manager
from core.file_manager import RawModuleExtractor

from core.qbrain_manager import QBrainTableManager

# Define Schemas
MODULE_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("file_type", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("binary_data", "BYTES", mode="NULLABLE"),
    bigquery.SchemaField("code", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("params", "STRING", mode="NULLABLE"),
]

SESSIONS_MODULES_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("module_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
]

MODULES_METHODS_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("module_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("method_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
]

def generate_numeric_id() -> str:
    """Generate a random numeric ID."""
    return str(random.randint(1000000000, 9999999999))

class ModuleWsManager(BQCore):
    DATASET_ID = "QBRAIN"
    MODULES_TABLE = "modules"
    SESSIONS_MODULES_TABLE = "sessions_to_modules"
    MODULES_METHODS_TABLE = "modules_to_methods"

    def __init__(self):
        super().__init__(dataset_id=self.DATASET_ID)
        self.qb = QBrainTableManager()
        self.table_ref = f"{self.MODULES_TABLE}"
        self.session_link_ref = f"{self.SESSIONS_MODULES_TABLE}"
        self.module_creator = RawModuleExtractor()

    def _ensure_module_table(self):
        """Check if modules table exists, create if not."""
        table_ref = f"{self.pid}.{self.DATASET_ID}.{self.MODULES_TABLE}"
        try:
            self.bqclient.get_table(table_ref)
            logging.info(f"Table {table_ref} exists.")
            # Ensure params column exists
            self.insert_col(self.MODULES_TABLE, "params", "STRING")
        except Exception as e:
            print(f"Error ensuring module table: {e}")
            logging.info(f"Creating table {table_ref}.")
            table = bigquery.Table(table_ref, schema=MODULE_SCHEMA)
            self.bqclient.create_table(table)
            logging.info(f"Table {table_ref} created.")

    def _ensure_sessions_modules_table(self):
        """Check if sessions_to_modules table exists, create if not."""
        table_ref = f"{self.pid}.{self.DATASET_ID}.{self.SESSIONS_MODULES_TABLE}"
        try:
            self.bqclient.get_table(table_ref)
            logging.info(f"Table {table_ref} exists.")
        except Exception as e:
            print(f"Error ensuring sessions_modules table: {e}")
            logging.info(f"Creating table {table_ref}.")
            table = bigquery.Table(table_ref, schema=SESSIONS_MODULES_SCHEMA)
            self.bqclient.create_table(table)
            logging.info(f"Table {table_ref} created.")

    def _ensure_modules_methods_table(self):
        """Check if modules_to_methods table exists, create if not."""
        table_ref = f"{self.pid}.{self.DATASET_ID}.{self.MODULES_METHODS_TABLE}"
        try:
            self.bqclient.get_table(table_ref)
            logging.info(f"Table {table_ref} exists.")
        except Exception as e:
            print(f"Error ensuring modules_methods table: {e}")
            logging.info(f"Creating table {table_ref}.")
            table = bigquery.Table(table_ref, schema=MODULES_METHODS_SCHEMA)
            self.bqclient.create_table(table)
            logging.info(f"Table {table_ref} created.")

    def set_module(self, rows: List[Dict] or Dict, user_id: str):
        if isinstance(rows, dict):
            rows = [rows]

        now = datetime.now().isoformat()
        for row in rows:
            row["user_id"] = user_id

            row["created_at"] = now
            row["updated_at"] = now

            # parent isn't in standard list, so serialize if present
            if row.get("parent"):
                row.pop("parent")

            if row.get("module_index"):
                row.pop("module_index")

        print("set module rows", len(rows))
        self.qb.set_item(self.MODULES_TABLE, rows)

    def link_session_module(self, session_id: str, module_id: str, user_id: str):
        """Link a module to a session."""
        row_id = generate_numeric_id()
        row = {
            "id": row_id,
            "session_id": session_id,
            "module_id": module_id,
            "user_id": user_id,
        }
        self.qb.set_item(self.SESSIONS_MODULES_TABLE, row) #, keys={"id": row["id"]}

    def link_module_methods(self, module_id: str, method_ids: List[str], user_id: str):
        """Link methods to a module."""
        # First, soft delete existing links for this module to avoid duplicates/stale links
        query = f"""
            UPDATE `{self.pid}.{self.DATASET_ID}.{self.MODULES_METHODS_TABLE}`
            SET status = 'deleted'
            WHERE module_id = @module_id AND user_id = @user_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        self.run_query(query, job_config=job_config)

        # Upsert new links
        rows = []
        for mid in method_ids:
            row_id = generate_numeric_id()
            rows.append({
                "id": row_id,
                "module_id": module_id,
                "method_id": mid,
                "user_id": user_id,
                "status": "active"
            })
        
        if rows:
            self.qb.set_item(self.MODULES_METHODS_TABLE, rows)

    def delete_module(self, module_id: str, user_id: str):
        """Delete module and its links."""
        # Delete from modules (Soft Delete)
        self.qb.del_entry(
            nid=module_id,
            table=self.MODULES_TABLE,
            user_id=user_id
        )

        # Delete from sessions_to_modules (Soft Delete)
        query2 = f"""
            UPDATE `{self.pid}.{self.DATASET_ID}.{self.SESSIONS_MODULES_TABLE}`
            SET status = 'deleted'
            WHERE module_id = @module_id AND user_id = @user_id
        """

        job_config2 = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        self.run_query(query2, job_config=job_config2, conv_to_dict=True)

        query3 = f"""
            UPDATE `{self.pid}.{self.DATASET_ID}.{self.MODULES_METHODS_TABLE}`
            SET status = 'deleted'
            WHERE module_id = @module_id AND user_id = @user_id
        """

        job_config3 = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        self.run_query(query3, job_config=job_config3, conv_to_dict=True)


    def rm_link_session_module(self, session_id: str, module_id: str, user_id: str):
        """Remove link session module (soft delete)."""
        self.qb.rm_link_session_link(
            session_id=session_id,
            nid=module_id,
            user_id=user_id,
            session_link_table=self.session_link_ref,
            session_to_link_name_id="module_id"
        )


    def update_module_params(self, module_id: str, user_id: str, params: Dict[str, Any] = None):
        """
        Update params field of a module.
        """
        if params is None:
            params = {}

        self.qb.set_item(
            self.MODULES_TABLE, 
            {"params": params}, 
            keys={"id": module_id, "user_id": user_id}
        )

    def retrieve_user_modules(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all modules for a user."""
        # todocreate sm user specific so user can edit it
        result = self.qb.get_users_entries(
            user_id=user_id,
            table=self.table_ref
        )
        
        modules = []
        for row in result:
            row_dict = dict(row)
            if row_dict.get("binary_data"):
                row_dict["binary_data"] = base64.b64encode(row_dict["binary_data"]).decode('utf-8')
            modules.append(row_dict)
        return modules

    def retrieve_session_modules(self, session_id: str, user_id: str, select: str = "m.*") -> List[Dict[str, Any]]:
        """Retrieve modules for a session."""
        # get sessions_to_modules rows -> get module rows
        links = self.qb.list_session_entries(
            user_id=user_id,
            session_id=session_id,
            table=self.session_link_ref,
            select="module_id"
        )
        
        module_ids = [row['module_id'] for row in links]
        
        # Now get the modules
        # Reuse get_module_by_id logic or call row_from_id directly
        # get_module_by_id returns {"modules": [...]}, we want just the list here or consistent return
        
        result = self.qb.row_from_id(
            nid=module_ids,
            select="*",
            table=self.table_ref
        )

        modules = []
        for row in result:
            row_dict = dict(row)
            # Process binary data if needed?
            # "return: data={modules:list[modules-table rows]}"
            # We usually shouldn't send bytes directly in JSON.
            # I will encode binary_data to base64 string if present
            if row_dict.get("binary_data"):
                row_dict["binary_data"] = base64.b64encode(row_dict["binary_data"]).decode('utf-8')
            modules.append(row_dict)
        return modules

    def get_module_by_id(
            self, module_id: str or list, select: str = "*") -> Optional[Dict[str, Any]]:
        """Get a single module by ID."""
        if isinstance(module_id, str):
            module_id = [module_id]

        rows = self.qb.row_from_id(
            nid=module_id,
            select=select,
            table=self.MODULES_TABLE
        )
        return {"modules": rows}


    def get_modules_fields(self, user_id: str, session_id: str, select: str = "f.*") -> Dict[str, Any]:
        """
        Get fields associated with modules in a session.
        """
        # Use FieldsManager to get IDs
        field_ids = fields_manager.retrieve_session_fields(session_id, user_id)
        
        if not field_ids:
            return {"fields": []}
            
        # Get full field objects
        # fields_manager.get_fields_by_id returns {"fields": [...]}
        response = fields_manager.get_fields_by_id(field_ids, select="*")
        return response



# Instantiate
module_manager = ModuleWsManager()

# -- RELAY HANDLERS --

def handle_list_users_modules(payload):
    """
    receive "LIST_USERS_MODULES": auth=user_id:str -> SEND_USERS_MODULES
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    if not user_id:
        return {"error": "Missing user_id"}
    
    modules = module_manager.retrieve_user_modules(user_id)
    # Return full objects now that they are safe (binary data encoded)
    
    return {
        "type": "LIST_USERS_MODULES", 
        "data": {"modules": modules} 
    }

def handle_send_sessions_modules(payload):
    # This seems to be the routine for returning session modules
    # But the user defined "receive GET_SESSIONS_MODULES" -> call this logic
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    
    if not user_id or not session_id:
        return {"error": "Missing user_id or session_id"}

    modules = module_manager.retrieve_session_modules(session_id, user_id)
    return {
        "type": "GET_SESSIONS_MODULES", # As per instruction
        "data": {"modules": modules}
    }

def handle_get_sessions_modules(payload):
    return handle_send_sessions_modules(payload)

def handle_link_session_module(payload):
    """
    receive "LINK_SESSION_MODULE": auth={user_id:str, module_id:str, session_id:str}
    -> LINK_SESSION_MODULE -> SEND_SESSIONS_MODULES
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    module_id = auth.get("module_id")
    session_id = auth.get("session_id")
    
    if not all([user_id, module_id, session_id]):
        return {"error": "Missing required auth fields"}
        
    module_manager.link_session_module(session_id, module_id, user_id)
    return handle_send_sessions_modules(payload)

def handle_rm_link_session_module(payload):
    """
    receive type="RM_LINK_SESSION_MODULE", auth={user_id:str, session_id:str, module_id:str}
    -> update sessions_to_modules-table row with status = deleted -> SEND_SESSION_MODULES
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    module_id = auth.get("module_id")
    
    if not all([user_id, session_id, module_id]):
        return {"error": "Missing required auth fields"}
        
    module_manager.rm_link_session_module(session_id, module_id, user_id)
    return handle_send_sessions_modules(payload)

def handle_del_module(payload):
    """
    receive "DEL_MODULE". auh={module_id:str, user_id:str}
    -> delete -> SEND_USERS_MODULES
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    module_id = auth.get("module_id") # User said 'module_id' in auth
    
    if not user_id or not module_id:
        return {"error": "Missing user_id or module_id"}
        
    module_manager.delete_module(module_id, user_id)
    return handle_list_users_modules(payload)

def handle_set_module(payload):
    """
    receive "SET_MODULE": data={files:list[files]}, auth={user_id:str, module_id:str, session_id:str}
    -> insert -> SEND_USERS_MODULES
    """
    auth = payload.get("auth", {})
    data = payload.get("data", {})
    user_id = auth.get("user_id")
    fields = data.get("fields", [])
    methods = data.get("methods", [])
    description = data.get("description", [])
    
    if not user_id:
        return {"error": "Missing user_id"}

    row = dict(
        id=data.get("id"),
        user_id=user_id,
        fields=fields,
        methods=methods,
        description=description,
        status="active",
    )

    original_id = auth.get("original_id")
    if original_id:
        module_manager.delete_module(original_id, user_id)

    module_manager.set_module(row, user_id)
    return handle_list_users_modules(payload)

def handle_get_module(payload):
    """
    receive "GET_MODULE": auth={module_id:str}
    -> get row -> convert -> return {type: "GET_MODULE", data:{id, file, code}}
    """
    auth = payload.get("auth", {})
    module_id = auth.get("module_id")
    
    if not module_id:
        return {"error": "Missing module_id"}
        
    row = module_manager.get_module_by_id(module_id)
    if not row:
        return {"error": "Module not found"}


    return {
        "type": "GET_MODULE",
        "data": row
    }



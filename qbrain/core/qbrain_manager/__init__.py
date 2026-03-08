"""
QBRAIN Table Manager

Centralized table management for the entire QBRAIN dataset.
Handles schema definitions and table creation for all QBRAIN tables.
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import dotenv

from qbrain._db.manager import get_db_manager, DBManager
from qbrain._db import queries as db_queries
from google import genai

dotenv.load_dotenv()

_QBRAIN_DEBUG = "[QBrainTableManager]"


class QBrainTableManager:
    """
    Centralized manager for all QBRAIN dataset tables.
    Handles schema definitions and table creation/verification.
    Receives BQCore instance via constructor (no inheritance).
    """

    DATASET_ID = "QBRAIN"

    def __init__(self, db:DBManager):
        self.db = db
        # Use plain table names for DuckDB (LOCAL_DB=True); prefixed for BigQuery
        self._local = os.getenv("LOCAL") or os.getenv("LOCAL_DB", "True") == "True"
        self.pid = os.getenv("PID") or "QBRAIN"
        self._table_ref = lambda t: t if self._local else f"{self.pid}.{self.DATASET_ID}.{t}"

        _gemini_key = os.environ.get("GEMINI_API_KEY")
        self.genai_client = genai.Client(api_key=_gemini_key) if _gemini_key else None
        print(f"{_QBRAIN_DEBUG} initialized with dataset: {self.DATASET_ID}")

    MANAGERS_INFO = [
        {
            "manager_name": "QBrainTableManager",
            "description": "Centralized table management for the entire QBRAIN dataset. Handles schema definitions and table creation/verification.",
            "default_table": None,
            "schema": None
        },
        {
            "manager_name": "UserManager",
            "description": "Manages user data, records, payments, and standard stack initialization in BigQuery.",
            "default_table": "users",
            "schema": {
                "id": "STRING",
                "email": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
                "status": "STRING",
                "sm_stack_status": "STRING"
            },
            "additional_tables": [
                {
                    "table_name": "payment",
                    "schema": {
                        "id": "STRING",
                        "uid": "STRING",
                        "payment_type": "STRING",
                        "created_at": "TIMESTAMP",
                        "updated_at": "TIMESTAMP",
                        "stripe_customer_id": "STRING",
                        "stripe_subscription_id": "STRING",
                        "stripe_payment_intent_id": "STRING",
                        "stripe_payment_method_id": "STRING"
                    }
                }
            ]
        },
        {
            "manager_name": "SessionManager",
            "description": "Manages user sessions, their lifecycles, and hierarchal links to environments, modules, and injections.",
            "default_table": "sessions",
            "schema": {
                "id": "STRING",
                "user_id": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
                "is_active": "STRING",
                "last_activity": "TIMESTAMP",
                "status": "STRING",
                "corpus_id": "STRING",
            }
        },
        {
            "manager_name": "EnvManager",
            "description": "Manages simulation environments, including their creation, deletion, model associations, and retrieval of linked data.",
            "default_table": "envs",
            "schema": {
                "id": "STRING",
                "user_id": "STRING",
                "sim_time": "STRING",
                "cluster_dim": "STRING",
                "dims": "STRING",
                "data": "STRING",
                "logs": "STRING",
                "description": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
                "status": "STRING",
                "pattern": "STRING",
                "model": "STRING",
            }
        },
        {
            "manager_name": "ModuleWsManager",
            "description": "Manages system modules, including upload, retrieval, and linking to sessions and methods.",
            "default_table": "modules",
            "schema": {
                "id": "STRING",
                "user_id": "STRING",
                "files": "STRING",
                "file_type": "STRING",
                "description": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
                "status": "STRING",
                "binary_data": "STRING",
                "methods": "STRING",
                "fields": "STRING",
                "origin": "STRING",
            }
        },
        {
            "manager_name": "FieldsManager",
            "description": "Manages fields (data containers), their parameters, and interaction links between fields.",
            "default_table": "fields",
            "schema": {
                "id": "STRING",
                "keys": "STRING",
                "user_id": "STRING",
                "module_id": "STRING",
                "status": "STRING",
                "origin": "STRING",
                "description": "STRING",
                "interactant_fields": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
            }
        },
        {
            "manager_name": "MethodManager",
            "description": "Manages computational methods, equations, and code snippets, including JAX generation and execution.",
            "default_table": "methods",
            "schema": {
                "id": "STRING",
                "user_id": "STRING",
                "description": "STRING",
                "equation": "STRING",
                "status": "STRING",
                "params": "STRING",
                "return_key": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
                "code": "STRING",
                "module_id": "STRING",
            }
        },
        {
            "manager_name": "ParamsManager",
            "description": "Manages global and local parameters used by methods and fields, including standard model constants.",
            "default_table": "params",
            "schema": {
                "id": "STRING",
                "param_type": "STRING",
                "user_id": "STRING",
                "axis_def": "STRING",
                "description": "STRING",
                "status": "STRING",
                "shape": "STRING",
                "value": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
            }
        },
        {
            "manager_name": "InjectionManager",
            "description": "Manages energy injection patterns and their association with environments and sessions.",
            "default_table": "injections",
            "schema": {
                "id": "STRING",
                "user_id": "STRING",  
                "data": "STRING",
                "frequency": "STRING",
                "amplitude": "STRING",
                "waveform": "STRING",
                "description": "STRING",
                "status": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
            }
        },
        {
            "manager_name": "FileManager",
            "description": "Handles file processing, extraction of code/configs (using RawModuleExtractor), and automated upsert of extracted entities.",
            "default_table": None,
            "table_name": "files",
            "schema": {
                "id": "STRING",
                "user_id": "STRING",
                "module_id": "STRING",
                "created_at": "TIMESTAMP",
                "rag_file_id": "STRING",
            },
        },
        {
            "manager_name": "ModelManager",
            "description": "Manages AI model interactions, specifically querying models associated with environments.",
             "default_table": None,
             "schema": None
        },
        {
            "manager_name": "Thalamus",
            "description": "Manages orchestration of simulations using Hopfield Network dynamics for high-level coordination and optimization.",
             "default_table": None,
             "schema": None
        },
        {
            "manager_name": "SMManager",
            "description": "Manages the Standard Model (SM) stack, initializing default graph structures and linking them to user environments.",
             "default_table": None,
             "schema": None
        },
        {
            "manager_name": "DataManager",
            "description": "Handles raw data conversion and interaction with real-time databases and visualization/training actors.",
             "default_table": None,
             "schema": None
        }
        ,
        {
            "manager_name": "PathfinderManager",
            "description": "Manages time-controller structs and recurring event metadata derived from JAX GTM runs for offline analysis.",
            "default_table": "controllers",
            "schema": {
                "id": "STRING",
                "env_id": "STRING",
                "user_id": "STRING",
                "time_ctlr": "STRING",
                "event_type": "STRING",
                "event_signature": "STRING",
                "meta": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
                "status": "STRING",
            },
        },
        {
            "manager_name": "GoalsManager",
            "description": "Manages goal configs (target env cfg) for simulation result analysis and parameter adaptation.",
            "default_table": "goals",
            "schema": {
                "id": "STRING",
                "user_id": "STRING",
                "goal_id": "STRING",
                "env_id": "STRING",
                "target_cfg": "STRING",
                "status": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
            },
        }
    ]
    
    @property
    def TABLES_SCHEMA(self):
        """
        Dynamically construct TABLES_SCHEMA from MANAGERS_INFO.
        """
        schemas = {}
        for manager in self.MANAGERS_INFO:
            if manager.get("default_table") and manager.get("schema"):
                schemas[manager["default_table"]] = manager["schema"]
            
            if manager.get("additional_tables"):
                for table_def in manager["additional_tables"]:
                    schemas[table_def["table_name"]] = table_def["schema"]
        return schemas

    # ------------------------------------------------------------------
    # Lightweight per-table/manager schema sync helpers
    # ------------------------------------------------------------------

    def ensure_table(self, table_name: str) -> Dict[str, str]:
        """
        Ensure a table exists in the current dataset (DuckDB or BigQuery).

        - If a schema is provided, it is used as-is.
        - Else, the schema is looked up in TABLES_SCHEMA.
        - If still unknown, a minimal default schema is used: {"id": "STRING"}.
        """
        try:
            table_item: dict = next(
                item
                for item in self.MANAGERS_INFO
                if table_name == item["default_table"]
            )
            if table_item:
                schema_sql = self.db.create_sql_schema(table_item["schema"])
                self.db.create_table(table_name, schema_sql)
            print(f"{_QBRAIN_DEBUG} ensure_table: table_name={table_name}")
        except Exception as e:
            print(f"{_QBRAIN_DEBUG} ensure_table: error for {table_name}: {e}")


    def ensure_manager_tables(self, manager_name: str) -> None:
        """
        Ensure default and additional tables for a given manager exist.

        manager_name must match MANAGERS_INFO.manager_name (e.g. 'EnvManager', 'ParamsManager').
        """
        for item in self.MANAGERS_INFO:
            try:

                default_table = item.get("default_table")
                default_schema = item.get("schema") or None
                if default_table:
                    self.ensure_table(default_table, default_schema)

                for tbl in item.get("additional_tables") or []:
                    tname = tbl.get("table_name")
                    tschema = tbl.get("schema") or None
                    if tname:
                        self.ensure_table(tname, tschema)
            except Exception as e:
                print(f"{_QBRAIN_DEBUG} ensure_manager_tables: error for {manager_name}: {e}")
                import traceback
                traceback.print_exc()

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Google GenAI.
        """
        if not self.genai_client:
            return []
            
        try:
            # Using text-embedding-004 as standard embedding model
            result = self.genai_client.models.embed_content(
                model="text-embedding-004",
                contents=text,
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def _generate_description(self, item: Dict[str, Any], table_name: str) -> str:
        """
        Generate a short description for an item using Google GenAI.
        """
        if not self.genai_client:
            return ""

        try:
            # Find manager info for context
            manager_desc = "Unknown Table"
            for info in self.MANAGERS_INFO:
                if info.get("default_table") == table_name:
                    manager_desc = info.get("description", "")
                    break
            
            # Construct prompt
            item_str = json.dumps({k: v for k, v in item.items() if k not in ["embedding", "binary_data"]}, default=str)
            prompt = f"""
            Generate a concise description (max 300 chars) for this database item.
            
            Context:
            Table: {table_name}
            Table Purpose: {manager_desc}
            
            Item Data:
            {item_str}
            
            Description:
            """
            
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()[:300]
        except Exception as e:
            print(f"Error generating description: {e}")
            return ""

    def _key_type_struct_to_json_schema(self, struct: Dict[str, Any], data_key: str = "items") -> Dict[str, Any]:
        """
        Convert key:type struct to STRING Schema for Gemini.
        Maps str/string -> STRING, int/float/number -> NUMBER, list/array -> ARRAY, dict/object -> OBJECT.
        """
        def _dict_to_item_schema(d: Dict[str, Any]) -> Dict[str, Any]:
            props = {}
            for k, v in (d or {}).items():
                if isinstance(v, dict):
                    props[k] = {"type": "object", "properties": _dict_to_item_schema(v).get("properties", {}), "additionalProperties": True}
                else:
                    desc = str(v) if isinstance(v, str) else ""
                    t = str(v).strip().lower() if v is not None else "string"
                    if t in ("int", "float", "number", "integer"):
                        props[k] = {"type": "number", "description": desc}
                    elif t in ("list", "array"):
                        props[k] = {"type": "array", "items": {"type": "string"}, "description": desc}
                    elif t in ("dict", "object"):
                        props[k] = {"type": "object", "additionalProperties": True, "description": desc}
                    else:
                        props[k] = {"type": "string", "description": desc}
            return {"type": "object", "properties": props, "additionalProperties": True}

        item_schema = _dict_to_item_schema(struct)
        return {
            "type": "object",
            "description": "Extracted items matching schema",
            "properties": {
                data_key: {
                    "type": "array",
                    "description": "Extracted items",
                    "items": item_schema,
                },
            },
            "required": [data_key],
        }

    def _schema_to_genai(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert STRING Schema type values to GenAI uppercase (OBJECT, ARRAY, STRING, NUMBER)."""
        if not isinstance(schema, dict):
            return schema
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.upper()
            elif k == "properties" and isinstance(v, dict):
                out[k] = {pk: self._schema_to_genai(pv) for pk, pv in v.items()}
            elif k == "items" and isinstance(v, dict):
                out[k] = self._schema_to_genai(v)
            else:
                out[k] = self._schema_to_genai(v) if isinstance(v, dict) else v
        return out

    def intelligent_extraction(
        self,
        prompt: str,
        content: str | bytes,
        schema_source: str | Dict[str, Any],
        manager_prompt_ext: str = "",
        table_name: Optional[str] = None,
        user_id: Optional[str] = None,
        data_key: str = "items",
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """
        Extract structured content using Gemini with schema from table param or dict.
        Returns raw parsed payload for manager.intelligent_processor.
        """
        if not self.genai_client:
            from qbrain.gem_core.gem import Gem
            gem = Gem()
            use_gem = True
        else:
            gem = None
            use_gem = False

        # Resolve schema from schema_source
        struct = None
        if isinstance(schema_source, dict):
            struct = schema_source
        elif isinstance(schema_source, str):
            rows = self.row_from_id([schema_source], "params", user_id=user_id)
            if rows:
                row = rows[0]
                for field in ("shape", "axis_def", "description"):
                    val = row.get(field)
                    if isinstance(val, str):
                        try:
                            struct = json.loads(val)
                            if isinstance(struct, dict):
                                break
                        except json.STRINGDecodeError:
                            pass
                    elif isinstance(val, dict):
                        struct = val
                        break
            if struct is None:
                struct = {}

        if struct is None:
            struct = {}

        json_schema = self._key_type_struct_to_json_schema(struct, data_key=data_key)
        json_schema = self._schema_to_genai(json_schema)

        content_str = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
        full_prompt = f"""{manager_prompt_ext}

        User instructions: {prompt or 'Extract all relevant content.'}
        
        Content:
        {content_str}
        
        Generate a STRING object with a "{data_key}" array. Each item must match the schema exactly.
        Return only valid STRING, no markdown or extra text."""

        config = {
            "response_mime_type": "application/json",
            "response_json_schema": json_schema,
        }

        try:
            if use_gem and gem:
                response = gem.ask(full_prompt, config=config)
            else:
                response = self.genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt,
                    config=config,
                )
                response = response.text if hasattr(response, "text") else str(response)

            text = (response or "").strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            items = parsed.get(data_key, parsed) if isinstance(parsed, dict) else parsed
            result = items if isinstance(items, list) else [items] if items else []
            print(f"{_QBRAIN_DEBUG} intelligent_extraction: done -> {len(result)} item(s)")
            return result
        except Exception as e:
            print(f"{_QBRAIN_DEBUG} intelligent_extraction error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def intelligent_extract_and_save(
        self,
        manager_name: str,
        prompt: str,
        content: str | bytes,
        user_id: str,
        schema_source: Optional[str | Dict[str, Any]] = None,
        **manager_kwargs,
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """
        Unified entry: extract via intelligent_extraction, process via manager.intelligent_processor, persist via set_item.
        manager_name: "param", "field", or "method".
        """
        manager_map = {
            "param": ("get_param_manager", "param", "items"),
            "field": ("get_field_manager", "field", "items"),
            "method": ("get_method_manager", "method", "methods"),
        }
        spec = manager_map.get(manager_name.lower())
        if not spec:
            print(f"{_QBRAIN_DEBUG} intelligent_extract_and_save: unknown manager {manager_name}")
            return []

        _getter_name, _content_key, data_key = spec
        try:
            from qbrain.core.managers_context import get_param_manager, get_field_manager, get_method_manager
            getters = {"param": get_param_manager, "field": get_field_manager, "method": get_method_manager}
            manager = getters[manager_name.lower()]()
        except Exception as e:
            print(f"{_QBRAIN_DEBUG} intelligent_extract_and_save: manager resolve error: {e}")
            return []

        schema = schema_source
        if schema is None and hasattr(manager, "get_schema_for_extraction"):
            schema = manager.get_schema_for_extraction()

        prompt_ext = ""
        if manager_name.lower() == "param":
            users_params = manager_kwargs.get("users_params", manager.get_users_params(user_id))
            prompt_ext = manager.extract_prompt(prompt, str(content)[:5000], users_params)
        elif manager_name.lower() == "field":
            params = manager_kwargs.get("params", [])
            fallback = manager_kwargs.get("fallback_params", [])
            prompt_ext = manager.extract_prompt(prompt, params, fallback)
        elif manager_name.lower() == "method":
            params = manager_kwargs.get("params", [])
            fallback = manager_kwargs.get("fallback_params", [])
            prompt_ext = manager.extract_prompt(params, fallback, prompt)

        raw = self.intelligent_extraction(
            prompt=prompt,
            content=content,
            schema_source=schema or {},
            manager_prompt_ext=prompt_ext,
            user_id=user_id,
            data_key=data_key,
        )
        if not raw:
            return []

        items = manager.intelligent_processor(raw, user_id=user_id, **manager_kwargs)
        if not items:
            return []

        if manager_name.lower() == "param":
            manager.set_param(items, user_id)
        elif manager_name.lower() == "field":
            manager.set_field(items, user_id)
        elif manager_name.lower() == "method":
            manager.set_method(items, user_id)

        return items

    def get_users_entries(self, user_id: str, table, select: str = "*"):
        """
        Retrieve all modules for a user.
        Groups by ID and returns only the newest entry per ID (based on created_at).
        """
        try:
            query, params = db_queries.duck_get_users_entries(
                table=self._table_ref(table),
                user_id=user_id,
                select=select,
            )

            #
            result = self.db.run_query(sql=query, params=params, conv_to_dict=True)

            # filter out deleted entries
            result = [entry for entry in result if entry["status"] != "deleted"]
            print("users params: ", result)
            return result
        except Exception as e:
            print(f"Error in get_users_entries: {e}")

    def list_session_entries(self, user_id: str, session_id:str, table:str, select: str = "*", partition_key: str = "id") -> Dict[str, list]:
        """
        Retrieve entries linked to session.
        Returns only the newest entry per ID (based on created_at).
        """
        print("list_session_entries")

        # Note: We filter status AFTER retrieval/row_numbering to ensure 'deleted' rows supersede 'active' rows.
        # Then we filter out the deleted ones from the result set.

        query, params = db_queries.duck_list_session_entries(
            table=self._table_ref(table),
            user_id=user_id,
            session_id=str(session_id),
            select=select,
            partition_key=partition_key,
        )
        session_rows = self.db.run_query(query, conv_to_dict=True, params=params)

        # filters out deleted
        session_rows = [entry for entry in session_rows if entry.get("status") != "deleted"]
        return session_rows

    
    def get_envs_linked_rows(
            self, 
            env_id: str, 
            user_id: str, 
            table_name: str,
            linked_row_id: str,
            linked_row_id_name:str,
            select: str = "*"
        ):
        query, params = db_queries.duck_get_envs_linked_rows(
            table=self._table_ref(table_name),
            env_id=env_id,
            linked_row_id=linked_row_id,
            linked_row_id_name=linked_row_id_name,
            user_id=user_id,
            select=select,
        )
        envs_linked_rows = self.db.run_query(query, conv_to_dict=True, params=params)
        
        envs_linked_rows = [entry for entry in envs_linked_rows if entry.get("status") != "deleted"]
        return envs_linked_rows
            

    def get_modules_linked_rows(
            self, 
            module_id: str, 
            user_id: str, 
            table_name: str,
            linked_row_id: str,
            linked_row_id_name:str,
            select: str = "*"
        ):
        print("get_modules_linked_rows")
        query, params = db_queries.duck_get_modules_linked_rows(
            table=self._table_ref(table_name),
            module_id=module_id,
            linked_row_id=linked_row_id,
            linked_row_id_name=linked_row_id_name,
            user_id=user_id,
            select=select,
        )
        modules_linked_rows = self.db.run_query(query, conv_to_dict=True, params=params)

        modules_linked_rows = [entry for entry in modules_linked_rows if entry.get("status") != "deleted"]
        return modules_linked_rows
        

    def row_from_id(self, nid:list or str, table, select="*", user_id=None):
        print("retrieve_env_from_id...", nid)
        if isinstance(nid, str):
            nid = [nid]

        query, params = db_queries.duck_row_from_id(
            table=self._table_ref(table),
            ids=nid,
            select=select,
            user_id=user_id,
        )

        items = self.db.run_query(query, params=params, conv_to_dict=True)
        #print("row_from_id: ", items)
        return items


    def initialize_all_tables(self) -> Dict[str, Any]:
        """
        Main workflow to initialize QBRAIN dataset and all tables.
        
        This should be called once at server startup to ensure all
        required tables exist with correct schemas.
        
        Returns:
            Dictionary containing initialization results
        """
        print("\n" + "=" * 70)
        print("INITIALIZING QBRAIN DATABASE TABLES")
        print("=" * 70)
        
        results = {
            "dataset_created": False,
            "tables_created": [],
            "tables_verified": [],
            "errors": []
        }
        
        try:
            # Step 1: Ensure dataset exists
            print("\n[1/2] Checking QBRAIN dataset...")
            results["dataset_created"] = self._ensure_dataset()
            
            # Step 2: Ensure all tables exist with correct schemas
            print("\n[2/2] Checking/creating all tables...")
            created, verified = self._ensure_all_tables()
            results["tables_created"] = created
            results["tables_verified"] = verified
            
            print("\n" + "=" * 70)
            print("[OK] QBRAIN TABLES SUCCESSFULLY INITIALIZED")
            print("=" * 70)
            print(f"Dataset: {self.DATASET_ID}")
            print(f"Tables created: {len(created)}")
            print(f"Tables verified: {len(verified)}")
            print(f"Total tables: {len(created) + len(verified)}")
            print("=" * 70 + "\n")
            
        except Exception as e:
            error_msg = f"Error initializing QBRAIN tables: {e}"
            # Avoid non-ASCII symbols so logs work in cp1252 consoles
            print(f"\n[ERROR] {error_msg}")
            results["errors"].append(error_msg)
            import traceback
            traceback.print_exc()
            raise
        
        return results

    def _ensure_dataset(self) -> bool:
        """
        Check if QBRAIN dataset exists, create if it doesn't.
        No-op for DuckDB (local).
        """
        try:
            if self._local:
                print(f"[OK] Dataset '{self.DATASET_ID}' ready (DuckDB)")
                return True
            self.bqcore.ensure_dataset_exists(self.DATASET_ID)
            print(f"[OK] Dataset '{self.DATASET_ID}' ready")
            return True
        except Exception as e:
            print(f"[ERROR] Error ensuring dataset: {e}")
            raise
            
    def _ensure_all_tables(self) -> tuple[List[str], List[str]]:
        """
        Check and create all required tables in QBRAIN dataset.
        
        Returns:
            Tuple of (newly_created_tables, already_existing_tables)
        """
        created_tables = []
        verified_tables = []
        
        total = len(self.TABLES_SCHEMA)
        for idx, (table_name, schema) in enumerate(self.TABLES_SCHEMA.items(), 1):
            try:
                print(f"\n  [{idx}/{total}] Processing table: {table_name}")
                
                # Check if table exists
                table_exists = self._table_exists(table_name)

                if table_exists:
                    verified_tables.append(table_name)
                    print(f"      [OK] Table '{table_name}' verified")

                else:
                    print(f"      Creating table '{table_name}'...")

                    self.db._duck_get_table_schema(
                        table_id=table_name,
                        schema=schema,
                        create_if_not_exists=True,
                    )
                    created_tables.append(table_name)

                    print(f"      [OK] Table '{table_name}' created")
                    
            except Exception as e:
                print(f"      [ERROR] Error with table {table_name}: {e}")
                raise
        
        return created_tables, verified_tables

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the dataset."""
        try:
            if self._local:
                con = self.db.connection
                r = con.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
                    [table_name],
                ).fetchone()
                return r is not None
            table_ref = f"{self.pid}.{self.DATASET_ID}.{table_name}"
            self.bqclient.get_table(table_ref)
            return True
        except Exception:
            return False

    def get_table_schema(
        self,
        table_id: str,
        schema: Optional[Dict[str, str]] = None,
        create_if_not_exists: bool = True,
    ) -> Dict[str, str]:
        """
        Ensure table exists with schema; return current schema.
        Delegates to DBManager for DuckDB (column logic stays in _db).
        """
        if self._local:
            return self.db._duck_get_table_schema(
                table_id=table_id,
                schema=schema or {},
                create_if_not_exists=create_if_not_exists,
            )
        # BigQuery path would go here if needed
        return schema or {}

    def get_schema(self, table_name: str) -> Dict[str, str]:
        """
        Get schema definition for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Schema dictionary for the table
        """
        return self.TABLES_SCHEMA.get(table_name, {})

    def list_all_tables(self) -> List[str]:
        """
        Get list of all defined table names.
        
        Returns:
            List of table names in TABLES_SCHEMA
        """
        return list(self.TABLES_SCHEMA.keys())



    def get_managers_info(self) -> List[Dict[str, str]]:
        """
        Get information about all manager classes.
        """
        return self.MANAGERS_INFO


    def set_item(
            self,
            table_name: str,
            items: Dict[str, Any] or list[dict],
    ) -> bool:
        """
        Universal method to insert or update an item.
        Serializes specific fields that are stored as STRING strings.
        If keys are provided, attempts to find and update existing row (via upsert_copy logic).
        If no keys or row not found, inserts new row.
        """
        print(f"set_item {len(items)} to", table_name)
        self.ensure_table(table_name)

        try:
            if isinstance(items, dict):
                items = [items]

            for item in items:
                if "parent" in item:
                    item.pop("parent")

                # Fields that must be strings in BQ but are often handled as dicts in code
                for key in item.keys():
                    # Check if the table_name and key exist in TABLES_SCHEMA before accessing
                    if table_name in self.TABLES_SCHEMA and key in self.TABLES_SCHEMA[table_name]:
                        if self.TABLES_SCHEMA[table_name][key] == "STRING" and not isinstance(item[key], str) and item[key] is not None:
                            try:
                                item[key] = json.dumps(item[key])
                            except Exception as e:
                                print(f"Warning: Failed to serialize field {key}: {e}")

                # 5. Timestamps & Status
                now = datetime.now()
                if "created_at" not in item:
                    item["created_at"] = now
                item["updated_at"] = now
                if "status" not in item:
                    item["status"] = "active"
        except Exception as e:
            print(f"Error in set_item: {e}")
            pass

        # Fallback to insert
        print("insert", table_name)
        self.db.insert(
            table_name,
            rows=items,
        )
        return True


    def reset_tables(self, table_list: List[str]):
        """
        Resets specified tables by recreating them with the defined schema.
        Uses BQGroundZero.DEFAULT_TIMESTAMP for created_at and updated_at.
        """
        print(f"Resetting tables: {table_list}")
        try:
            for table_name in table_list:
                if table_name not in self.TABLES_SCHEMA:
                    print(f"Skipping unknown table: {table_name}")
                    continue

                schema = self.TABLES_SCHEMA[table_name]

                # Construct columns definition
                cols_def = []
                default_ts = "CURRENT_TIMESTAMP"
                for col, dtype in schema.items():
                    col_def = f"{col} {dtype}"
                    if col in ["created_at", "updated_at"]:
                        col_def += f" DEFAULT {default_ts}"
                    cols_def.append(col_def)

                cols_str = ",\n  ".join(cols_def)

                query = f"CREATE OR REPLACE TABLE {table_name} ({cols_str})"

                print(f"Recreating table {table_name}...")
                try:
                    self.db.run_query(query)
                    print(f"Table {table_name} reset.")
                except Exception as e:
                    print(f"Error resetting table {table_name}: {e}")

        except Exception as e:
            print(f"Error in set_item: {e}")
            return False


# Default singleton for standalone use (no orchestrator context)
_default_bqcore = None
_qbrain_table_manager_instance: Optional["QBrainTableManager"] = None
LOCAL_DB:str = os.getenv("LOCAL_DB", "True")

def get_qbrain_table_manager(bqcore=None) -> "QBrainTableManager":
    """Return QBrainTableManager. Uses BigQuery when LOCAL_DB=False and bqcore given; else DuckDB."""
    global _qbrain_table_manager_instance, _default_bqcore

    if _qbrain_table_manager_instance is None:
        db = get_db_manager()
        _qbrain_table_manager_instance = QBrainTableManager(
            db=db,

        )
    return _qbrain_table_manager_instance


if __name__ == "__main__":
    qbrain_manager = get_qbrain_table_manager()
    qbrain_manager.reset_tables(["fields"])

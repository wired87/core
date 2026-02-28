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

from _db.manager import get_db_manager
from utils.str_size import get_str_size
from google import genai

dotenv.load_dotenv()

from google.cloud import bigquery

_QBRAIN_DEBUG = "[QBrainTableManager]"


class QBrainTableManager:
    """
    Centralized manager for all QBRAIN dataset tables.
    Handles schema definitions and table creation/verification.
    Receives BQCore instance via constructor (no inheritance).
    """

    DATASET_ID = "QBRAIN"

    def __init__(self, db):
        self.db = db
        self.insert_col = db.insert_col
        self.pid = db.pid
        self.ds_ref = db.ds_ref
        self._local = db.local
        self.duck_con = db.connection
        self.bqcore = db.bqcore
        self.bqclient = getattr(db, "bqclient", None)
        self.run_db = db.run_db
        self.run_query = db.run_query
        self.get_table_schema = db.get_table_schema
        self._table_ref = lambda t: t if db.local else f"{db.pid}.{self.DATASET_ID}.{t}"
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
                "id": "INTEGER",
                "user_id": "STRING",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
                "is_active": "BOOLEAN",
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
                "sim_time": "INTEGER",
                "cluster_dim": "INTEGER",
                "dims": "INTEGER",
                "data": "STRING",
                "logs": "JSON",
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
                "files": "JSON",
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
                "embedding": "ARRAY<FLOAT64>",
                "status": "STRING",
                "shape": "STRING",
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
            "schema": None,
            "additional_tables": [
                {
                    "table_name": "files",
                    "schema": {
                        "id": "STRING",
                        "user_id": "STRING",
                        "module_id": "STRING",
                        "created_at": "TIMESTAMP",
                        "rag_file_id": "STRING",
                    }
                }
            ]
        },
        {
            "manager_name": "ModelManager",
            "description": "Manages AI model interactions, specifically querying models associated with environments.",
             "default_table": None,
             "schema": None
        },
        {
            "manager_name": "OrchestratorManager",
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

    def get_users_entries(self, user_id: str, table, select: str = "*"):
        """
        Retrieve all modules for a user.
        Groups by ID and returns only the newest entry per ID (based on created_at).
        """
        job_config=None
        try:
            query = f"""
                    SELECT {select}
                    FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
                        FROM {self._table_ref(table)}
                        WHERE (user_id = @user_id OR user_id = 'public') AND (status != 'deleted' OR status IS NOT NULL)
                    )
                    WHERE row_num = 1
                """

            if self.db.local is True:
                query = sqlglot.transpile(
                    query,
                    read="bigquery",
                    write="duckdb"
                )[0]
            else:
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                    ]
                )
            result = self.db.run_query(
                sql=query,
                params={"user_id": user_id},
                job_config=job_config,
                conv_to_dict=True
            )

            # filter out deleted entries
            result = [entry for entry in result if entry["status"] != "deleted"]
            return result
        except Exception as e:
            print(f"Error in get_users_entries: {e}")

    def list_session_entries(self, user_id: str, session_id:str, table:str, select: str = "*", partition_key: str = "id") -> Dict[str, list]:
        """
        Retrieve entries linked to session.
        Returns only the newest entry per ID (based on created_at).
        """
        print("list_session_entries")

        query = f"""
            SELECT {select}, status
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_key} ORDER BY created_at DESC) as row_num
                FROM {self._table_ref(table)}
                WHERE user_id = @user_id AND session_id = @session_id
            )
            WHERE row_num = 1
        """
        # Note: We filter status AFTER retrieval/row_numbering to ensure 'deleted' rows supersede 'active' rows.
        # Then we filter out the deleted ones from the result set.
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("session_id", "STRING", session_id)
            ]
        )
        session_rows = self.run_query(query, conv_to_dict=True, job_config=job_config)
        
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
        query = f"""
            SELECT {select}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
                FROM {self._table_ref(table_name)}
                WHERE env_id = @env_id AND {linked_row_id_name} = @{linked_row_id_name} AND user_id = @user_id AND (status != 'deleted' OR status IS NULL)
            )
            WHERE row_num = 1
        """

        job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("env_id", "STRING", env_id),
                    bigquery.ScalarQueryParameter(linked_row_id_name, "STRING", linked_row_id),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                ]
            )
            
        envs_linked_rows = self.run_query(query, conv_to_dict=True, job_config=job_config)
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

        query = f"""
            SELECT {select}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
                FROM {self._table_ref(table_name)}
                WHERE module_id = @module_id AND {linked_row_id_name} = @{linked_row_id_name} AND user_id = @user_id AND (status != 'deleted' OR status IS NULL)
            )
            WHERE row_num = 1
        """

        job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
                    bigquery.ScalarQueryParameter(linked_row_id_name, "STRING", linked_row_id),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                ]
            )
            
        modules_linked_rows = self.run_query(query, conv_to_dict=True, job_config=job_config)
        modules_linked_rows = [entry for entry in modules_linked_rows if entry.get("status") != "deleted"]
        return modules_linked_rows
        

    def row_from_id(self, nid:list or str, table, select="*", user_id=None):
        print("retrieve_env_from_id...", nid)
        if isinstance(nid, str):
            nid = [nid]
        
        if self._local:
            # DuckDB: use list param with unnest
            tbl = self._table_ref(table)
            id_placeholders = ", ".join(["?"] * len(nid))
            user_filter = " AND user_id = ?" if user_id else ""
            params = list(nid)
            
            if user_id:
                params.append(user_id)
            query = f"""
                SELECT {select}
                FROM (
                    SELECT {select}, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
                    FROM {tbl}
                    WHERE id IN ({id_placeholders}) AND (status != 'deleted' OR status IS NOT NULL){user_filter}
                )
                WHERE row_num = 1
            """
            items = self.db.run_query(query, params=params, conv_to_dict=True)
        else:
            query = f"""
                SELECT {select}
                FROM (
                    SELECT {select}, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
                    FROM {tbl}
                    WHERE id IN UNNEST(@id) AND (status != 'deleted' OR status IS NOT NULL)
                )
                WHERE row_num = 1
            """
            query_parameters = [bigquery.ArrayQueryParameter("id", "STRING", nid)]
            if user_id:
                query += " AND user_id = @user_id"
                query_parameters.append(bigquery.ScalarQueryParameter("user_id", "STRING", user_id))
            job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
            items = self.run_query(query, conv_to_dict=True, job_config=job_config)
        return items


    def upsert_copy(self, table_name: str, keys: Dict[str, Any], updates: Dict[str, Any]) -> bool:
        """
        Fetches the latest row matching keys, updates it with `updates`, and inserts as new row.
        This avoids 'streaming buffer' errors on UPDATE/DELETE operations.
        """
        print("upsert_copy...")
        try:
            # Construct SELECT query
            where_clause = " AND ".join([f"{k} = @{k}" for k in keys.keys()])

            ref = f"{self.pid}.{self.DATASET_ID}.{table_name}"
            # Check if table_name is actually a ref passed from caller
            if "." in table_name:
                ref = table_name
                # extract clean table name for bq_insert if needed
                clean_table_name = table_name.split(".")[-1]
            else:
                clean_table_name = table_name

            # Sort by created_at DESC to get latest version
            query = f"""
                SELECT * FROM `{ref}` 
                WHERE {where_clause} 
                ORDER BY created_at DESC LIMIT 1
            """
            # Use schema types for params (sessions.id is INT64, not STRING)
            schema = self.TABLES_SCHEMA.get(clean_table_name, {})
            if self._local:
                # DuckDB: use table name only, no project.dataset prefix
                duck_query = f"SELECT * FROM {clean_table_name} WHERE " + " AND ".join([f"{k} = ?" for k in keys.keys()]) + " ORDER BY created_at DESC LIMIT 1"
                ordered = [int(v) if schema.get(k) in ("INTEGER", "INT64") else str(v) for k, v in keys.items()]
                rows = self.db.run_db(duck_query, conv_to_dict=True, params=ordered)
            else:
                bq_params = []
                for k, v in keys.items():
                    col_type = schema.get(k, "STRING")
                    if col_type in ("INTEGER", "INT64"):
                        bq_params.append(bigquery.ScalarQueryParameter(k, "INT64", int(v)))
                    else:
                        bq_params.append(bigquery.ScalarQueryParameter(k, "STRING", str(v)))
                job_config = bigquery.QueryJobConfig(query_parameters=bq_params)
                rows = list(self.bqclient.query(query, job_config=job_config).result())
            if not rows:
                print(f"Row not found for upsert_copy in {ref} with keys {keys}")
                return False
                
            row = dict(rows[0])
            row.update(updates)

            # check injection
            if "data" in row and not isinstance(row["data"], str):
                row["data"] = json.dumps(row["data"])


            # check injection
            if "keys" in row and not isinstance(row["keys"], str):
                row["keys"] = json.dumps(row["keys"])

            # check field
            if "values" in row and not isinstance(row["values"], str):
                row["values"] = json.dumps(row["values"])

            # check field
            if "axis_def" in row and not isinstance(row["axis_def"], str):
                row["axis_def"] = json.dumps(row["axis_def"])

            # check params
            if "params" in row and not isinstance(row["params"], str):
                row["params"] = json.dumps(row["params"])

            # Update timestamps for the new version
            now = datetime.now().isoformat()
            row["created_at"] = now
            if "updated_at" in row:
                row["updated_at"] = now

            if "created_at" in row:
                row["created_at"] = now
                
            # Re-insert
            self.db.insert(clean_table_name, [row])
            return True
            
        except Exception as e:
            print(f"Error in upsert_copy for {table_name}: {e}")
            return False

    def del_entry(self, nid: str, table:str, user_id:str, name_id:str="id") -> bool:
        """
        Soft Delete an entry from a table.
        Uses upsert_copy.
        """
        # Determine strict ID or internal ID? 
        keys = {name_id: nid, "user_id": user_id}
        updates = {"status": "deleted"}
        
        # Determine table_name from ref to pass to upsert_copy
        # Ref is typically `project.dataset.table`
        return self.upsert_copy(table, keys, updates)

    def rm_env_link(
            self,
            env_id: str,
            injection_id: str,
            user_id: str,
            table_name: str,
            linked_row_id: str,
            linked_row_id_name: str,
        ):
        
        keys = {
            "env_id": env_id,
            linked_row_id_name: injection_id,
            "user_id": user_id
        }
        updates = {"status": "deleted"}
        
        success = self.upsert_copy(table_name, keys, updates)
        if success:
             print(f"Removed link for injection {injection_id} from environment {env_id}")
        return success


    def rm_module_link(
            self,
            module_id: str,
            linked_id: str,
            user_id: str,
            table_name: str,
            linked_row_id_name: str,
        ):
        
        keys = {
            "module_id": module_id,
            linked_row_id_name: linked_id,
            "user_id": user_id
        }
        updates = {"status": "deleted"}
        
        success = self.upsert_copy(table_name, keys, updates)
        if success:
             print(f"Removed link for module {module_id}")
        return success


    def rm_link_session_link(
            self,
            session_id: str,
            nid: str,
            user_id: str,
            session_link_table:str,
            session_to_link_name_id:str
    ):
        """Remove link env to session (soft delete via upsert_copy)."""
        keys = {
            "session_id": session_id,
            session_to_link_name_id: nid,
            "user_id": user_id
        }
        updates = {"status": "deleted"}
        return self.upsert_copy(session_link_table, keys, updates)




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
            print("✓ QBRAIN TABLES SUCCESSFULLY INITIALIZED")
            print("=" * 70)
            print(f"Dataset: {self.DATASET_ID}")
            print(f"Tables created: {len(created)}")
            print(f"Tables verified: {len(verified)}")
            print(f"Total tables: {len(created) + len(verified)}")
            print("=" * 70 + "\n")
            
        except Exception as e:
            error_msg = f"Error initializing QBRAIN tables: {e}"
            print(f"\n❌ {error_msg}")
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
                print(f"✓ Dataset '{self.DATASET_ID}' ready (DuckDB)")
                return True
            self.bqcore.ensure_dataset_exists(self.DATASET_ID)
            print(f"✓ Dataset '{self.DATASET_ID}' ready")
            return True
        except Exception as e:
            print(f"❌ Error ensuring dataset: {e}")
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
                    print(f"      ✓ Table '{table_name}' verified")
                    
                    current_schema = self.get_table_schema(
                        table_id=table_name,
                        schema=schema,
                        create_if_not_exists=True
                    )

                    missing_cols = [(cn, ct) for cn, ct in schema.items() if cn not in current_schema]
                    if missing_cols:
                        print(f"      ⚠ Found missing columns in '{table_name}': {missing_cols}")
                        for col_name, col_type in missing_cols:
                            try:
                                self.insert_col(table_name, col_name, col_type)
                                print(f"        ✓ Added column {col_name}")
                            except Exception as e:
                                print(f"        ❌ Failed to add column {col_name}: {e}")

                else:
                    print(f"      Creating table '{table_name}'...")
                    if self._local:
                        self.get_table_schema(table_id=table_name, schema=schema, create_if_not_exists=True)
                        created_tables.append(table_name)
                    else:
                        schema_list = [bigquery.SchemaField(cn, ct) for cn, ct in schema.items()]
                        table_ref = f"{self.pid}.{self.DATASET_ID}.{table_name}"
                        table = bigquery.Table(table_ref, schema=schema_list)
                        self.bqclient.create_table(table)
                        created_tables.append(table_name)
                    print(f"      ✓ Table '{table_name}' created")
                    
            except Exception as e:
                print(f"      ❌ Error with table {table_name}: {e}")
                raise
        
        return created_tables, verified_tables

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the dataset."""
        try:
            if self._local:
                r = self.duck_con.execute(f"SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone()
                return r is not None
            table_ref = f"{self.pid}.{self.DATASET_ID}.{table_name}"
            self.bqclient.get_table(table_ref)
            return True
        except Exception:
            return False

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


    def update_env_pattern(self, env_id: str, pattern_data: dict, user_id: str):
        """
        Updates the 'pattern' column for a specific environment in the 'envs' table.
        Uses upsert_copy to maintain history.
        """
        try:
            print(f"Update Env Pattern for {env_id}")
            table_name = "envs"
            keys = {"id": env_id, "user_id": user_id}

            # pattern_data is likely a dict, so we convert it to JSON string
            # BQ schema for 'pattern' is STRING.
            pattern_str = json.dumps(pattern_data)
            get_str_size(pattern_str)

            updates = {"pattern": pattern_str}

            success = self.upsert_copy(table_name, keys, updates)
            if success:
                print(f"Successfully updated pattern for env {env_id}")
            else:
                print(f"Failed to update pattern for env {env_id}")
            return success
        except Exception as e:
            print(f"Error updating pattern for env {env_id}", e)

    def set_item(
            self,
            table_name: str,
            items: Dict[str, Any] or list[dict],
            keys: Dict[str, Any] = None,
    ) -> bool:
        """
        Universal method to insert or update an item.
        Serializes specific fields that are stored as JSON strings.
        If keys are provided, attempts to find and update existing row (via upsert_copy logic).
        If no keys or row not found, inserts new row.
        """
        print("set_item...")
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
                now = datetime.now().isoformat()
                if "created_at" not in item:
                    item["created_at"] = now
                item["updated_at"] = now
                if "status" not in item:
                    item["status"] = "active"

            # 5. Update or Insert
            if keys:
                if self.upsert_copy(table_name, keys, items[0]):
                    return True
        except Exception as e:
            print(f"Error in set_item: {e}")
            pass
        
        # Fallback to insert
        print("insert", table_name)
        self.db.insert(table_name, rows=items)
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
                default_ts = "CURRENT_TIMESTAMP" if self._local else self.bqcore.DEFAULT_TIMESTAMP
                for col, dtype in schema.items():
                    col_def = f"{col} {dtype}"
                    if col in ["created_at", "updated_at"]:
                        col_def += f" DEFAULT {default_ts}"
                    cols_def.append(col_def)

                cols_str = ",\n  ".join(cols_def)
                if self._local:
                    query = f"CREATE OR REPLACE TABLE {table_name} ({cols_str})"
                else:
                    query = f"CREATE OR REPLACE TABLE `{self.pid}.{self.DATASET_ID}.{table_name}` (\n  {cols_str}\n)"

                print(f"Recreating table {table_name}...")
                try:
                    self.run_db(query)
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

import json
import logging
import random
from datetime import datetime
from typing import Dict, Any, List
from google.cloud import bigquery
from a_b_c.bq_agent._bq_core.bq_handler import BQCore

from core.qbrain_manager import QBrainTableManager

# Debug prefix for manager methods (grep-friendly)
_ENV_DEBUG = "[EnvManager]"

# Define Schema
ENV_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("sim_time", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("cluster_dim", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("dims", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"), # Link to users table
    bigquery.SchemaField("data", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP"),
    bigquery.SchemaField("updated_at", "TIMESTAMP"),
]

class EnvManager(BQCore):
    DATASET_ID = "QBRAIN"
    TABLE_ID = "envs"

    def __init__(self):
        super().__init__(dataset_id=self.DATASET_ID)
        self.qb = QBrainTableManager()
        self.session_link_tref = f"session_to_envs"

    def _ensure_env_table(self):
        """Check if envs table exists, create if not."""
        try:
            print(f"{_ENV_DEBUG} _ensure_env_table: checking table")
            table_ref = f"{self.pid}.{self.DATASET_ID}.{self.TABLE_ID}"
            self.bqclient.get_table(table_ref)
            logging.info(f"Table {table_ref} exists.")
            self.insert_col(self.TABLE_ID, "data", "STRING")
            print(f"{_ENV_DEBUG} _ensure_env_table: table exists, data column ensured")
        except Exception as e:
            print(f"{_ENV_DEBUG} _ensure_env_table: creating table: {e}")
            logging.info(f"Creating table {table_ref}:{e}")
            table = bigquery.Table(table_ref, schema=ENV_SCHEMA)
            self.bqclient.create_table(table)
            logging.info(f"Table {table_ref} created.")
            print(f"{_ENV_DEBUG} _ensure_env_table: table created")

    def retrieve_send_user_specific_env_table_rows(self, user_id: str, select: str = "*") -> Dict[str, Any]:
        """
        Retrieve envs linked to user_id.
        Returns data payload for 'get_env' message.
        """
        try:
            print(f"{_ENV_DEBUG} retrieve_send_user_specific_env_table_rows: user_id={user_id}")
            envs = self.qb.get_users_entries(
                user_id=user_id,
                table=self.TABLE_ID,
            )
            print(f"{_ENV_DEBUG} retrieve_send_user_specific_env_table_rows: got {len(envs) if envs else 0} envs")
            return {"envs": envs}
        except Exception as e:
            print(f"{_ENV_DEBUG} retrieve_send_user_specific_env_table_rows: error: {e}")
            logging.error(f"Error retrieving envs for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return {"envs": []}

    def retrieve_session_envs(self, user_id: str, session_id:str, select: str = "*") -> Dict[str, list]:
        """
        Retrieve envs linked to session.
        Returns data payload for 'get_env' message.
        """
        try:
            print(f"{_ENV_DEBUG} retrieve_session_envs: user_id={user_id}, session_id={session_id}")
            linked_env_rows = self.qb.list_session_entries(
                user_id,
                session_id,
                table=self.session_link_tref,
                select="env_id",
                partition_key="env_id"
            )
            env_ids = [row["env_id"] for row in linked_env_rows]
            print(f"{_ENV_DEBUG} retrieve_session_envs: env_ids={env_ids}")
            envs = self.retrieve_env_from_id(env_id=env_ids)
            return envs
        except Exception as e:
            print(f"{_ENV_DEBUG} retrieve_session_envs: error: {e}")
            logging.error(f"Error retrieving envs for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return {"envs": []}

    def retrieve_env_from_id(
            self,
            env_id: str or list[str],
            select: str = "*"
    ) -> Dict[str, Any]:
        """
        Retrieve env based on injection_id.
        Returns data payload for 'get_env' message.
        """
        try:
            print(f"{_ENV_DEBUG} retrieve_env_from_id: env_id={env_id}")
            envs = self.qb.row_from_id(
                nid=env_id,
                select=select,
                table=self.TABLE_ID
            )
            if envs:
                print(f"{_ENV_DEBUG} retrieve_env_from_id: got {len(envs)} env(s)")
                return {"envs": envs}
            return {"envs": []}
        except Exception as e:
            print(f"{_ENV_DEBUG} retrieve_env_from_id: error: {e}")
            logging.error(f"Error retrieving env: {env_id}: {e}")
            import traceback
            traceback.print_exc()
            return {"envs": []}


    def delete_env(self, env_id: str, user_id: str):
        """Delete env from table."""
        try:
            print(f"{_ENV_DEBUG} delete_env: env_id={env_id}, user_id={user_id}")
            self.qb.del_entry(
                nid=env_id,
                table=self.TABLE_ID,
                user_id=user_id,
            )
            print(f"{_ENV_DEBUG} delete_env: done")
        except Exception as e:
            print(f"{_ENV_DEBUG} delete_env: error: {e}")
            import traceback
            traceback.print_exc()
            raise


    def set_env(self, env_data: Dict[str, Any], user_id: str):
        """
        Insert user env.
        env_data should match env_item_type (id, sim_time, cluster_dim, dims)
        """
        try:
            print(f"{_ENV_DEBUG} set_env: user_id={user_id}, env_id={env_data.get('id')}")
            row = env_data.copy()
            row["user_id"] = user_id
            if "data" in row and isinstance(row["data"], (dict, list)):
                try:
                    row["data"] = json.dumps(row["data"])
                except Exception as e:
                    print(f"{_ENV_DEBUG} set_env: serializing data: {e}")
            self.qb.set_item(self.TABLE_ID, row, keys={"id": row.get("id"), "user_id": user_id})
            print(f"{_ENV_DEBUG} set_env: done")
        except Exception as e:
            print(f"{_ENV_DEBUG} set_env: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def link_session_env(self, session_id: str, env_id: str, user_id: str):
        """Link env to session."""
        try:
            print(f"{_ENV_DEBUG} link_session_env: session_id={session_id}, env_id={env_id}")
            row_id = str(random.randint(1000000000, 9999999999))
            row = {
                "id": row_id,
                "session_id": session_id,
                "env_id": env_id,
                "user_id": user_id,
            }
            self.qb.set_item("session_to_envs", row)
            print(f"{_ENV_DEBUG} link_session_env: done")
        except Exception as e:
            print(f"{_ENV_DEBUG} link_session_env: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def rm_link_session_env(self, session_id: str, env_id: str, user_id: str):
        """Remove link env to session (soft delete)."""
        try:
            print(f"{_ENV_DEBUG} rm_link_session_env: session_id={session_id}, env_id={env_id}")
            self.qb.rm_link_session_link(
                session_id=session_id,
                nid=env_id,
                user_id=user_id,
                session_link_table=self.session_link_tref,
                session_to_link_name_id="env_id",
            )
            print(f"{_ENV_DEBUG} rm_link_session_env: done")
        except Exception as e:
            print(f"{_ENV_DEBUG} rm_link_session_env: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def link_env_module(self, session_id: str, env_id: str, module_id: str, user_id: str):
        """Link module to env in a session."""
        try:
            print(f"{_ENV_DEBUG} link_env_module: session_id={session_id}, env_id={env_id}, module_id={module_id}")
            row_id = str(random.randint(1000000000, 9999999999))
            row = {
                "id": row_id,
                "session_id": session_id,
                "env_id": env_id,
                "module_id": module_id,
                "user_id": user_id,
            }
            self.qb.set_item("envs_to_modules", row)
            print(f"{_ENV_DEBUG} link_env_module: done")
        except Exception as e:
            print(f"{_ENV_DEBUG} link_env_module: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def rm_link_env_module(self, session_id: str, env_id: str, module_id: str, user_id: str):
        """Remove link module to env (soft delete)."""
        try:
            print(f"{_ENV_DEBUG} rm_link_env_module: session_id={session_id}, env_id={env_id}, module_id={module_id}")
            query = f"""
                UPDATE `{self.pid}.{self.DATASET_ID}.envs_to_modules`
                SET status = 'deleted'
                WHERE session_id = @session_id AND env_id = @env_id AND module_id = @module_id AND user_id = @user_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
                    bigquery.ScalarQueryParameter("env_id", "STRING", env_id),
                    bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                ]
            )
            self.run_query(query, job_config=job_config)
            print(f"{_ENV_DEBUG} rm_link_env_module: done")
        except Exception as e:
            print(f"{_ENV_DEBUG} rm_link_env_module: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_env_module_structure(self, session_id: str, env_id: str, user_id: str) -> Dict:
        """
        Get hierarchical structure for an env in a session.
        Returns: {sessions: {sid: {envs: {eid: {modules: {mid: {fields: []}}}}}}}
        """
        try:
            print(f"{_ENV_DEBUG} get_env_module_structure: session_id={session_id}, env_id={env_id}")
            ds = f"{self.pid}.{self.DATASET_ID}"
            # 1. Get Modules
            q_mods = f"""
            SELECT module_id FROM `{ds}.envs_to_modules`
            WHERE session_id=@sid AND env_id=@eid AND (status != 'deleted' OR status IS NULL)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("sid", "STRING", session_id),
                    bigquery.ScalarQueryParameter("eid", "STRING", env_id)
                ]
            )
            mod_rows = self.run_query(q_mods, conv_to_dict=True, job_config=job_config)
            module_ids = [r['module_id'] for r in mod_rows]

            modules_struct = {}

            if module_ids:
                # 2. Get Fields for these modules
                q_fields = f"""
                    SELECT module_id, field_id FROM `{ds}.modules_to_fields`
                    WHERE session_id=@sid AND env_id=@eid AND module_id IN UNNEST(@mids) 
                    AND (status != 'deleted' OR status IS NULL)
                """
                job_config_f = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("sid", "STRING", session_id),
                        bigquery.ScalarQueryParameter("eid", "STRING", env_id),
                        bigquery.ArrayQueryParameter("mids", "STRING", module_ids)
                    ]
                )
                field_rows = self.run_query(q_fields, conv_to_dict=True, job_config=job_config_f)

                # Group fields by module
                for mid in module_ids:
                    modules_struct[mid] = {"fields": []}

                for row in field_rows:
                    mid = row['module_id']
                    fid = row['field_id']
                    if mid in modules_struct:
                        modules_struct[mid]["fields"].append(fid)
                print(f"{_ENV_DEBUG} get_env_module_structure: done")
                return {
                    "sessions": {
                        session_id: {
                            "envs": {
                                env_id: {
                                    "modules": modules_struct
                                }
                            }
                        }
                    }
                }
        except Exception as e:
            print(f"{_ENV_DEBUG} get_env_module_structure: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def download_model(self, env_id: str, user_id: str) -> Dict[str, Any]:
        """
        Process logic to download model for env_id.
        """
        try:
            print(f"{_ENV_DEBUG} download_model: env_id={env_id}, user_id={user_id}")
            env = self.retrieve_env_from_id(env_id)
            if not env or not env.get("envs"):
                print(f"{_ENV_DEBUG} download_model: env not found")
                return {"error": "Env not found"}
            print(f"{_ENV_DEBUG} download_model: done")
            return {
                "env_id": env_id,
                "status": "ready",
                "download_url": f"https://storage.googleapis.com/models/{user_id}/{env_id}/model.zip"
            }
        except Exception as e:
            print(f"{_ENV_DEBUG} download_model: error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def retrieve_logs_env(self, env_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve logs for env_id.
        Expected return: [{"timestamp": "ISOString", "message": "Log message"}]
        """
        try:
            print(f"{_ENV_DEBUG} retrieve_logs_env: env_id={env_id}, user_id={user_id}")
            query = f"""
                SELECT timestamp, message 
                FROM `{self.pid}.{self.DATASET_ID}.logs`
                WHERE env_id = @env_id AND user_id = @user_id
                ORDER BY timestamp DESC
                LIMIT 100
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("env_id", "STRING", env_id),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                ]
            )
            rows = self.run_query(query, conv_to_dict=True, job_config=job_config)
            formatted_logs = []
            for row in rows:
                ts = row.get("timestamp")
                if hasattr(ts, 'isoformat'):
                    ts = ts.isoformat()
                formatted_logs.append({
                    "timestamp": str(ts),
                    "message": row.get("message", "")
                })
            print(f"{_ENV_DEBUG} retrieve_logs_env: got {len(formatted_logs)} log(s)")
            return formatted_logs
        except Exception as e:
            print(f"{_ENV_DEBUG} retrieve_logs_env: error (returning mock): {e}")
            import traceback
            traceback.print_exc()
            return [
                {"timestamp": datetime.now().isoformat(), "message": f"System: Log retrieval initialized for {env_id}."},
                {"timestamp": datetime.now().isoformat(), "message": "System: Waiting for simulation stream..."},
                {"timestamp": datetime.now().isoformat(), "message": f"Error: {e}"}
            ]

    def get_env_data(self, env_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve live env data (Bottom Live Data Table).
        Expected return: [{"key1": "value1", ...}, ...]
        """
        try:
            print(f"{_ENV_DEBUG} get_env_data: env_id={env_id}, user_id={user_id}")
            query = f"""
                SELECT *
                FROM `{self.pid}.{self.DATASET_ID}.sim_data`
                WHERE env_id = @env_id
                ORDER BY created_at DESC
                LIMIT 50
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("env_id", "STRING", env_id)
                ]
            )
            rows = self.run_query(query, conv_to_dict=True, job_config=job_config)
            print(f"{_ENV_DEBUG} get_env_data: got {len(rows)} row(s)")
            return rows
        except Exception as e:
            print(f"{_ENV_DEBUG} get_env_data: error (returning mock): {e}")
            import traceback
            traceback.print_exc()
            return [
                {"step": 1, "loss": 0.5, "metric_a": 10},
                {"step": 2, "loss": 0.4, "metric_a": 12},
                {"step": 3, "loss": 0.35, "metric_a": 15},
            ]


# Instantiate
print("Instantiating global env_manager")
env_manager = EnvManager()
print("Global env_manager instantiated")

def handle_get_env(payload):
    """
    receive "get_env": auth=(user_id:str) -> retireve_send...
    """
    print("handle_get_env")
    auth = payload["auth"]
    env_id = auth["env_id"]
    resp_data = env_manager.retrieve_env_from_id(env_id)
    
    return {
        "type": "GET_ENV",
        "data": resp_data,
    }


def handle_get_envs_session(payload):
    auth = payload["auth"]
    user_id = auth.get("user_id")





def handle_get_envs_user(payload):
    """
    receive "get_envs_user": auth
    Same as get_env
    """
    print("handle_get_envs_user")

    auth = payload["auth"]
    user_id = auth.get("user_id")

    if not user_id:
        return {"error": "No user_id in auth"}

    # Retrieve data
    resp_data = env_manager.retrieve_send_user_specific_env_table_rows(user_id)

    return {
        "type": "GET_USERS_ENVS",
        "data": resp_data
    }

def handle_del_env(payload):
    """
    receive "del_env": auth={env_id:str, user_id:str}
    """
    print("handle_del_env")

    auth = payload["auth"]
    user_id = auth.get("user_id")
    env_id = auth.get("env_id")
    
    if not user_id or not env_id:
        return {"error": "Missing env_id or user_id in auth"}
        
    env_manager.delete_env(env_id, user_id)

    # Retrieve data
    resp_data = env_manager.retrieve_send_user_specific_env_table_rows(user_id)

    # Return message structure.
    # Logic: send get_env: data:{envs: ...}
    return {
        "type": "GET_USERS_ENVS",
        "data": resp_data
    }

def handle_set_env(payload):
    """
    env_item_type=
    "env_item_type: {
            id: STRING,
            sim_time: INT64,
            cluster_dim: INT64,
            dims: INT64,
            user_id:STRING,
        }

    receive "set_env": data={env:env_item_type}, auth={user_id:str, session_id:str, }
    """
    print("handle_set_env")

    auth = payload["auth"]
    data = payload["data"]
    user_id = auth.get("user_id")
    env_data = data.get("env")
    
    if not user_id or not env_data:
        return {"error": "Missing user_id or env data"}
        
    original_id = auth.get("original_id")
    if original_id:
        env_manager.delete_env(original_id, user_id)

    env_manager.set_env(env_data, user_id)

    # Retrieve data
    resp_data = env_manager.retrieve_send_user_specific_env_table_rows(user_id)

    # Return message structure.
    # Logic: send get_env: data:{envs: ...}
    # Logic: send get_env: data:{envs: ...}
    return {
        "type": "GET_USERS_ENVS",
        "data": resp_data
    }

def handle_get_sessions_envs(payload):
    """
    receive "GET_SESSIONS_ENVS": auth={user_id:str, session_id:str}
    -> return envs...
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    
    if not user_id or not session_id:
        return {"error": "Missing user_id or session_id"}
        
    data = env_manager.retrieve_session_envs(user_id, session_id)
    return {
        "type": "GET_SESSIONS_ENVS",
        "data": data
    }

def handle_link_session_env(payload):
    """Link session to env."""
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    # Prompt says auth={user_id, session_id, env_id} but also data? Prompt: "auth={...}"
    session_id = auth.get("session_id")
    env_id = auth.get("env_id")
    
    if not all([user_id, session_id, env_id]):
        return {"error": "Missing required fields"}
        
    env_manager.link_session_env(session_id, env_id, user_id)
    return handle_get_sessions_envs(payload)

def handle_rm_link_session_env(payload):
    """Remove link session to env."""
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    env_id = auth.get("env_id")
    
    if not all([user_id, session_id, env_id]):
        return {"error": "Missing required fields"}
    
    # Remove the link
    env_manager.rm_link_session_env(session_id, env_id, user_id)
    
    # Return the remaining links structure
    from core.session_manager.session import session_manager
    structure = session_manager.get_full_session_structure(user_id, session_id)
    
    return {
        "type": "LIST_SESSIONS_ENVS",
        "data": structure
    }


def handle_link_env_module(payload):
    """Link module to env."""
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    env_id = auth.get("env_id")
    module_id = auth.get("module_id")

    if not all([user_id, session_id, env_id, module_id]):
        return {"error": "Missing required fields"}

    env_manager.link_env_module(session_id, env_id, module_id, user_id)
    
    structure = env_manager.get_env_module_structure(session_id, env_id, user_id)
    return {
        "type": "LINK_ENV_MODULE",
        "data": structure
    }

def handle_rm_link_env_module(payload):
    """Remove link module to env."""
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    env_id = auth.get("env_id")
    module_id = auth.get("module_id")

    if not all([user_id, session_id, env_id, module_id]):
        return {"error": "Missing required fields"}

    env_manager.rm_link_env_module(session_id, env_id, module_id, user_id)
    
    from core.session_manager.session import session_manager
    structure = session_manager.get_full_session_structure(user_id, session_id)
    
    return {
        "type": "LINK_ENV_MODULE", # Return updated structure
        "data": structure
    }


def handle_download_model(payload):
    """
    receive "DOWNLOAD_MODEL"
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    env_id = auth.get("env_id")

    if not user_id or not env_id:
        return {"error": "Missing user_id or env_id"}

    data = env_manager.download_model(env_id, user_id)
    
    # Response structure not strictly defined but usually involves sending data back.
    # We will send a type that expects the download url or confirmation.
    # Assuming "DOWNLOAD_MODEL" type back is fine.
    return {
        "type": "DOWNLOAD_MODEL",
        "data": data
    }


def handle_retrieve_logs_env(payload):
    """
    receive "RETRIEVE_LOGS_ENV"
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    env_id = auth.get("env_id")

    if not user_id or not env_id:
        return {"error": "Missing user_id or env_id"}

    logs = env_manager.retrieve_logs_env(env_id, user_id)
    
    return {
        "type": "RETRIEVE_LOGS_ENV",
        "data": logs
    }


def handle_get_env_data(payload):
    """
    receive "GET_ENV_DATA"
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    env_id = auth.get("env_id")

    if not user_id or not env_id:
        return {"error": "Missing user_id or env_id"}

    env_data = env_manager.get_env_data(env_id, user_id)
    
    return {
        "type": "GET_ENV_DATA",
        "data": env_data
    }




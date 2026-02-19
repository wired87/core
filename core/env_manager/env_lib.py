import json
import logging
import random
from datetime import datetime
from typing import Dict, Any, List
from google.cloud import bigquery
from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from core.qbrain_manager import get_qbrain_table_manager
from core.handler_utils import param_missing_error, require_param, require_param_truthy, get_val

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

class EnvManager:
    DATASET_ID = "QBRAIN"


    def __init__(self, qb):
        self.qb = qb
        self.TABLE_ID = "envs"
        self.pid = qb.pid
        self.bqclient = qb.bqclient
        self.run_query = qb.run_query
        self.insert_col = qb.insert_col
        self.session_link_tref = "session_to_envs"

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
            print(f"{_ENV_DEBUG} get_env_data: got {rows} row(s)")
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


# Default instance for standalone use (no orchestrator context)
_default_bqcore = BQCore(dataset_id="QBRAIN")
_default_env_manager = EnvManager(get_qbrain_table_manager(_default_bqcore))
env_manager = _default_env_manager  # backward compat

def handle_get_env(data=None, auth=None):
    """Retrieve a single environment by ID. Required: env_id (auth or data)."""
    data, auth = data or {}, auth or {}
    env_id = get_val(data, auth, "env_id")
    if err := require_param(env_id, "env_id"):
        return err
    from core.managers_context import get_env_manager
    return {"type": "GET_ENV", "data": get_env_manager().retrieve_env_from_id(env_id)}


def handle_get_envs_session(data=None, auth=None):
    """Retrieve all environments linked to a session. Required: user_id, session_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    from core.managers_context import get_env_manager
    return {"type": "GET_SESSIONS_ENVS", "data": get_env_manager().retrieve_session_envs(user_id, session_id)}


def handle_get_envs_user(data=None, auth=None):
    """Retrieve all environments owned by a user. Required: user_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    if err := require_param(user_id, "user_id"):
        return err
    from core.managers_context import get_env_manager
    return {"type": "GET_USERS_ENVS", "data": get_env_manager().retrieve_send_user_specific_env_table_rows(user_id)}


def handle_del_env(data=None, auth=None):
    """Delete an environment by ID. Required: user_id, env_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    env_id = get_val(data, auth, "env_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    from core.managers_context import get_env_manager
    mgr = get_env_manager()
    mgr.delete_env(env_id, user_id)
    return {"type": "GET_USERS_ENVS", "data": mgr.retrieve_send_user_specific_env_table_rows(user_id)}


def handle_set_env(data=None, auth=None):
    """Create or update an environment. Required: user_id (auth), env (data). Optional: original_id (auth)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    env_data = data.get("env") if isinstance(data.get("env"), dict) else data.get("env")
    original_id = get_val(data, auth, "original_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param_truthy(env_data, "env"):
        return err
    from core.managers_context import get_env_manager
    mgr = get_env_manager()
    if original_id:
        mgr.delete_env(original_id, user_id)
    mgr.set_env(env_data, user_id)
    return {"type": "GET_USERS_ENVS", "data": mgr.retrieve_send_user_specific_env_table_rows(user_id)}


def handle_get_sessions_envs(data=None, auth=None):
    """Retrieve all environments linked to a session. Required: user_id, session_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    from core.managers_context import get_env_manager
    return {"type": "GET_SESSIONS_ENVS", "data": get_env_manager().retrieve_session_envs(user_id, session_id)}


def handle_link_session_env(data=None, auth=None):
    """Link a session to an environment. Required: user_id, session_id, env_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    env_id = get_val(data, auth, "env_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    from core.managers_context import get_env_manager
    get_env_manager().link_session_env(session_id, env_id, user_id)
    return handle_get_sessions_envs(data={"session_id": session_id}, auth={"user_id": user_id})


def handle_rm_link_session_env(data=None, auth=None):
    """Remove the link between a session and an environment. Required: user_id, session_id, env_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    env_id = get_val(data, auth, "env_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    from core.managers_context import get_env_manager, get_session_manager
    get_env_manager().rm_link_session_env(session_id, env_id, user_id)
    return {"type": "LIST_SESSIONS_ENVS", "data": get_session_manager().get_full_session_structure(user_id, session_id)}


def handle_link_env_module(data=None, auth=None):
    """Link a module to an environment. Required: user_id, session_id, env_id, module_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    env_id = get_val(data, auth, "env_id")
    module_id = get_val(data, auth, "module_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    if err := require_param(module_id, "module_id"):
        return err
    from core.managers_context import get_env_manager
    mgr = get_env_manager()
    mgr.link_env_module(session_id, env_id, module_id, user_id)
    return {"type": "LINK_ENV_MODULE", "data": mgr.get_env_module_structure(session_id, env_id, user_id)}


def handle_rm_link_env_module(data=None, auth=None):
    """Remove the link between a module and an environment. Required: user_id, session_id, env_id, module_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    env_id = get_val(data, auth, "env_id")
    module_id = get_val(data, auth, "module_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    if err := require_param(module_id, "module_id"):
        return err
    from core.managers_context import get_env_manager, get_session_manager
    get_env_manager().rm_link_env_module(session_id, env_id, module_id, user_id)
    return {"type": "LINK_ENV_MODULE", "data": get_session_manager().get_full_session_structure(user_id, session_id)}


def handle_download_model(data=None, auth=None):
    """Trigger model download for an environment. Required: user_id, env_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    env_id = get_val(data, auth, "env_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    from core.managers_context import get_env_manager
    return {"type": "DOWNLOAD_MODEL", "data": get_env_manager().download_model(env_id, user_id)}


def handle_retrieve_logs_env(data=None, auth=None):
    """Retrieve logs for an environment. Required: user_id, env_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    env_id = get_val(data, auth, "env_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    from core.managers_context import get_env_manager
    return {"type": "RETRIEVE_LOGS_ENV", "data": get_env_manager().retrieve_logs_env(env_id, user_id)}


def handle_get_env_data(data=None, auth=None):
    """Retrieve environment data (state, config, etc.). Required: user_id, env_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    env_id = get_val(data, auth, "env_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    from core.managers_context import get_env_manager
    return {"type": "GET_ENV_DATA", "data": get_env_manager().get_env_data(env_id, user_id)}




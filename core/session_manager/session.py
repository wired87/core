import pprint
import random
from datetime import datetime
from typing import Optional, Dict, Any, List
from google.cloud import bigquery
import json

from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from core.env_manager.env_lib import env_manager
from core.module_manager.ws_modules_manager.modules_lib import module_manager
from core.qbrain_manager import QBrainTableManager

_SESSION_DEBUG = "[SessionManager]"


class SessionManager(BQCore):
    """
    Manages user sessions in BigQuery.
    Extends BQCore to leverage existing BigQuery functionality.
    """

    DATASET_ID = "QBRAIN"
    TABLE_ID="sessions"
    # Sessions table schema
    SESSIONS_TABLE_SCHEMA = {
        "id": "INT64",
        "user_id": "STRING",
        "created_at": "TIMESTAMP",
        "updated_at": "TIMESTAMP",
        "is_active": "BOOL",
        "last_activity": "TIMESTAMP",
        "research_files": "STRING",  # JSON list of URLs
        "corpus_id": "STRING",
    }

    # Link tables schemas
    SESSIONS_TO_ENVS_SCHEMA = [
        bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("env_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    ]

    SESSIONS_TO_INJECTIONS_SCHEMA = [
        bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("injection_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    ]

    def __init__(self):
        """Initialize SessionManager with QBRAIN dataset."""
        try:
            BQCore.__init__(self, dataset_id=self.DATASET_ID)
            self.ds_ref = f"{self.pid}.{self.DATASET_ID}"
            self.qb = QBrainTableManager()
            print(f"{_SESSION_DEBUG} initialized with dataset: {self.DATASET_ID}")
        except Exception as e:
            print(f"{_SESSION_DEBUG} __init__ error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _ensure_sessions_table(self) -> bool:
        """
        Check if sessions table exists, create if it doesn't.
        
        Returns:
            True if table was created or already exists
        """
        try:
            print(f"{_SESSION_DEBUG} _ensure_sessions_table: checking/creating")
            self.get_table_schema(
                table_id="sessions",
                schema=self.SESSIONS_TABLE_SCHEMA,
                create_if_not_exists=True
            )
            print(f"{_SESSION_DEBUG} _ensure_sessions_table: ready")
            return True
        except Exception as e:
            print(f"{_SESSION_DEBUG} _ensure_sessions_table: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _ensure_link_tables(self):
        """Ensure session link tables exist."""
        try:
            print(f"{_SESSION_DEBUG} _ensure_link_tables: ensuring session_to_envs, session_to_injections")
            env_link_schema = {field.name: field.field_type for field in self.SESSIONS_TO_ENVS_SCHEMA}
            self.get_table_schema(
                table_id="session_to_envs",
                schema=env_link_schema,
                create_if_not_exists=True
            )
            inj_link_schema = {field.name: field.field_type for field in self.SESSIONS_TO_INJECTIONS_SCHEMA}
            self.get_table_schema(
                table_id="session_to_injections",
                schema=inj_link_schema,
                create_if_not_exists=True
            )
            print(f"{_SESSION_DEBUG} _ensure_link_tables: done")
        except Exception as e:
            print(f"{_SESSION_DEBUG} _ensure_link_tables: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_session_id(self) -> int:
        """
        Generate a random numeric session ID.
        
        Returns:
            Random integer session ID (10-15 digits)
        """
        # Generate a random 10-15 digit number to ensure uniqueness
        return random.randint(10**12, 10**15 - 1)


    def create_session(self, user_id: str) -> Optional[int]:
        """
        Create a new session for a user after successful authentication.
        Upserts a session row with foreign key to the users table.
        
        Args:
            user_id: User unique identifier (must exist in users table)
            
        Returns:
            Session ID if successful, None if failed
        """
        print("create_session...")
        try:
            print(f"{_SESSION_DEBUG} create_session: user_id={user_id}")
            user_check_query = f"""
                SELECT uid FROM `{self.ds_ref}.users`
                WHERE uid = @user_id AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
            )

            result = self.bqclient.query(user_check_query, job_config=job_config).result()
            if result.total_rows == 0:
                print(f"{_SESSION_DEBUG} create_session: user does not exist")
                return None
            session_id = self._generate_session_id()
            while self._session_exists(session_id):
                session_id = self._generate_session_id()

            now = datetime.now().isoformat()

            session_data = {
                "id": session_id,
                "user_id": user_id,
                "created_at": now,
                "updated_at": now,
                "is_active": True,
                "last_activity": now,
                "status": "active",
                "research_files": "[]"
            }

            self.qb.set_item(
                "sessions",
                session_data,
                keys={"id": session_id}
            )
            print(f"{_SESSION_DEBUG} create_session: created session_id={session_id}")
            return session_id
        except Exception as e:
            print(f"{_SESSION_DEBUG} create_session: error: {e}")
            import traceback
            traceback.print_exc()
        print("create_session... done")


    def _session_exists(self, session_id: int) -> bool:
        """
        Check if a session ID already exists.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            True if session exists
        """
        try:
            query = f"""
                SELECT id FROM `{self.ds_ref}.sessions`
                WHERE id = @session_id AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("session_id", "INT64", session_id)]
            )
            result = self.bqclient.query(query, job_config=job_config).result()
            return result.total_rows > 0
        except Exception as e:
            print(f"{_SESSION_DEBUG} _session_exists: error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve session record by session ID.
        
        Args:
            session_id: Session unique identifier
            
        Returns:
            Session record as dictionary or None if not found
        """
        try:
            print(f"{_SESSION_DEBUG} get_session: session_id={session_id}")
            query = f"""
                SELECT * FROM `{self.ds_ref}.sessions`
                WHERE id = @session_id AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("session_id", "INT64", session_id)]
            )
            result = self.bqclient.query(query, job_config=job_config).result()
            for row in result:
                print(f"{_SESSION_DEBUG} get_session: found")
                return dict(row)
            print(f"{_SESSION_DEBUG} get_session: not found")
            return None
        except Exception as e:
            print(f"{_SESSION_DEBUG} get_session: error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all sessions where user_id matches the provided auth user_id.
        Sorts entries from new to old.
        """
        try:
            print(f"{_SESSION_DEBUG} list_user_sessions: user_id={user_id}")
            sessions = self.qb.get_users_entries(
                user_id=user_id,
                table=self.TABLE_ID,
            )
            print(f"{_SESSION_DEBUG} list_user_sessions: got {len(sessions) if sessions else 0} session(s)")
            return sessions
        except Exception as e:
            print(f"{_SESSION_DEBUG} list_user_sessions: error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_user_sessions(self, user_id: str, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve all sessions for a specific user.
        
        Args:
            user_id: User unique identifier
            active_only: If True, only return active sessions
            
        Returns:
            List of session records as dictionaries
        """
        try:
            print(f"{_SESSION_DEBUG} get_user_sessions: user_id={user_id}, active_only={active_only}")
            active_filter = "AND is_active = TRUE" if active_only else ""
            query = f"""
                SELECT * FROM `{self.ds_ref}.sessions`
                WHERE user_id = @user_id {active_filter} AND (status != 'deleted' OR status IS NULL)
                ORDER BY created_at DESC
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
            )
            result = self.bqclient.query(query, job_config=job_config).result()
            sessions = [dict(row) for row in result]
            print(f"{_SESSION_DEBUG} get_user_sessions: got {len(sessions)} session(s)")
            return sessions
        except Exception as e:
            print(f"{_SESSION_DEBUG} get_user_sessions: error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def deactivate_session(self, session_id: int) -> bool:
        """
        Deactivate a session by marking it as inactive.
        Uses FETCH + INSERT to avoid update streaming buffer limits.
        """
        try:
            print(f"{_SESSION_DEBUG} deactivate_session: session_id={session_id}")
            updates = {
                "is_active": False,
                "last_activity": datetime.now().isoformat(),
                "status": "deleted"
            }
            out = self.qb.set_item(
                "sessions",
                updates,
                keys={"id": session_id}
            )
            print(f"{_SESSION_DEBUG} deactivate_session: done, success={out}")
            return out
        except Exception as e:
            print(f"{_SESSION_DEBUG} deactivate_session: error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_session_activity(self, session_id: int) -> bool:
        """
        Update the last_activity timestamp for a session.
        Uses FETCH + INSERT to avoid update streaming buffer limits.
        """
        try:
            print(f"{_SESSION_DEBUG} update_session_activity: session_id={session_id}")
            updates = {"last_activity": datetime.utcnow().isoformat()}
            return self.qb.set_item("sessions", updates, keys={"id": session_id})
        except Exception as e:
            print(f"{_SESSION_DEBUG} update_session_activity: error: {e}")
            import traceback
            traceback.print_exc()
            return False


    def get_session_modules(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get modules for a session.
        Uses module_manager to retrieve data.
        """
        try:
            print(f"{_SESSION_DEBUG} get_session_modules: user_id={user_id}, session_id={session_id}")
            modules = module_manager.retrieve_session_modules(session_id, user_id)
            print(f"{_SESSION_DEBUG} get_session_modules: got {len(modules) if modules else 0} module(s)")
            return {"modules": modules}
        except Exception as e:
            print(f"{_SESSION_DEBUG} get_session_modules: error: {e}")
            import traceback
            traceback.print_exc()
            return {"modules": []}

    def get_session_envs(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get environments for a session.
        """
        try:
            query = f"""
            SELECT env_id, user_id, status
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY env_id ORDER BY created_at DESC) as row_num
                FROM `{self.ds_ref}.session_to_envs`
                WHERE user_id = @user_id AND session_id = @session_id
            )
            WHERE row_num = 1 AND status != 'deleted'
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("session_id", "STRING", str(session_id)),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                ]
            )
            
            result = self.run_query(query, job_config=job_config)
            
            env_ids = [entry.get("env_id") for entry in result if entry.get("status") != "deleted"]

            if not env_ids:
                return {"envs": []}

            envs = env_manager.retrieve_env_from_id(env_ids)
            
            return envs
        except Exception as e:
            print(f"{_SESSION_DEBUG} get_session_envs: error: {e}")
            import traceback
            traceback.print_exc()
            return {"envs": []}

    def link_env_session(self, user_id: str, session_id: str, env_id: str):
        """Link env to session."""
        try:
            print(f"{_SESSION_DEBUG} link_env_session: user_id={user_id}, session_id={session_id}, env_id={env_id}")
            row_id = str(random.randint(1000000000, 9999999999))
            row = {
                "id": row_id,
                "session_id": str(session_id),
                "env_id": env_id,
                "user_id": user_id,
            }
            self.qb.set_item("session_to_envs", row)
            print(f"{_SESSION_DEBUG} link_env_session: done")
        except Exception as e:
            print(f"{_SESSION_DEBUG} link_env_session: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def rm_link_env_session(self, user_id: str, session_id: str, env_id: str):
        """
        Remove link env to session (soft delete).
        Uses INSERT instead of UPDATE to avoid BigQuery streaming buffer limitations.
        """
        try:
            print(f"{_SESSION_DEBUG} rm_link_env_session: user_id={user_id}, session_id={session_id}, env_id={env_id}")
            self.qb.rm_link_session_link(
                session_id=session_id,
                nid=env_id,
                user_id=user_id,
                session_link_table="session_to_envs",
                session_to_link_name_id="env_id"
            )
            print(f"{_SESSION_DEBUG} rm_link_env_session: done")
        except Exception as e:
            print(f"{_SESSION_DEBUG} rm_link_env_session: error: {e}")
            import traceback
            traceback.print_exc()
            raise 



    def get_active_session(self, user_id: str) -> Optional[int]:
        """
        Get the most recent active session for a user.
        """
        try:
            print(f"{_SESSION_DEBUG} get_active_session: user_id={user_id}")
            query = f"""
                SELECT id FROM `{self.ds_ref}.sessions`
                WHERE user_id = @user_id AND is_active = TRUE AND status != 'deleted'
                ORDER BY created_at DESC
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
            )
            result = self.run_query(query, conv_to_dict=True, job_config=job_config)
            if result:
                print(f"{_SESSION_DEBUG} get_active_session: found session_id={result[0]['id']}")
                return result[0]["id"]
            print(f"{_SESSION_DEBUG} get_active_session: no active session")
            return None
        except Exception as e:
            print(f"{_SESSION_DEBUG} get_active_session: error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_or_create_active_session(self, user_id: str) -> Optional[int]:
        """
        Get existing active session or create a new one.
        Minimizes session creation per connection.
        """
        try:
            print(f"{_SESSION_DEBUG} get_or_create_active_session: user_id={user_id}")
            session_id = self.get_active_session(user_id)
            if session_id:
                print(f"{_SESSION_DEBUG} get_or_create_active_session: using existing session_id={session_id}")
                return session_id

            # Create a new session and rely on the returned ID instead of
            # immediately querying BigQuery again (which can be eventually consistent).
            new_session_id = self.create_session(user_id)
            if not new_session_id:
                print(f"{_SESSION_DEBUG} get_or_create_active_session: failed to create session for user_id={user_id}")
                return None

            print(f"{_SESSION_DEBUG} get_or_create_active_session: created, session_id={new_session_id}")
            return new_session_id
        except Exception as e:
            print(f"{_SESSION_DEBUG} get_or_create_active_session: error: {e}")
            import traceback
            traceback.print_exc()
            return None
            

    def get_full_session_structure(self, user_id: str, session_id: str) -> Dict:
        """
        Retrieve hierarchical structure of session -> env -> modules -> fields
        for ALL active environments in the session.
        """
        try:
            print(f"{_SESSION_DEBUG} get_full_session_structure: user_id={user_id}, session_id={session_id}")
            ds = f"{self.pid}.{self.DATASET_ID}"

            q_envs = f"""
            SELECT env_id FROM `{ds}.session_to_envs`
            WHERE session_id=@sid AND user_id=@uid AND (status != 'deleted' OR status IS NULL)
            """
            job_config_e = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("sid", "STRING", session_id),
                    bigquery.ScalarQueryParameter("uid", "STRING", user_id)
                ]
            )
            env_rows = self.run_query(q_envs, conv_to_dict=True, job_config=job_config_e)
            active_env_ids = [r['env_id'] for r in env_rows]

            if not active_env_ids:
                return {
                    "sessions": {
                        session_id: {
                            "envs": {}
                        }
                    }
                }

            # 2. Get Modules for these envs
            q_mods = f"""
                SELECT env_id, module_id FROM `{ds}.envs_to_modules`
                WHERE session_id=@sid AND env_id IN UNNEST(@eids) AND (status != 'deleted' OR status IS NULL)
            """
            job_config_m = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("sid", "STRING", session_id),
                    bigquery.ArrayQueryParameter("eids", "STRING", active_env_ids)
                ]
            )
            mod_rows = self.run_query(q_mods, conv_to_dict=True, job_config=job_config_m)

            # Map env_id -> list of module_ids
            env_modules_map = {eid: [] for eid in active_env_ids}
            all_module_ids = []
            for r in mod_rows:
                eid = r['env_id']
                mid = r['module_id']
                if eid in env_modules_map:
                    env_modules_map[eid].append(mid)
                all_module_ids.append(mid)

            modules_struct_by_env = {} # env_id -> { module_id: { fields: [] } }

            # Pre-initialize structures
            for eid in active_env_ids:
                modules_struct_by_env[eid] = {}
                for mid in env_modules_map[eid]:
                    modules_struct_by_env[eid][mid] = {"fields": []}

            if all_module_ids:
                # 3. Get Fields for these modules
                q_fields = f"""
                    SELECT env_id, module_id, field_id FROM `{ds}.modules_to_fields`
                    WHERE session_id=@sid AND env_id IN UNNEST(@eids) AND module_id IN UNNEST(@mids) 
                    AND (status != 'deleted' OR status IS NULL)
                """
                job_config_f = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("sid", "STRING", session_id),
                        bigquery.ArrayQueryParameter("eids", "STRING", active_env_ids),
                        bigquery.ArrayQueryParameter("mids", "STRING", all_module_ids)
                    ]
                )
                field_rows = self.run_query(q_fields, conv_to_dict=True, job_config=job_config_f)

                for row in field_rows:
                    eid = row['env_id']
                    mid = row['module_id']
                    fid = row['field_id']

                    if eid in modules_struct_by_env and mid in modules_struct_by_env[eid]:
                         modules_struct_by_env[eid][mid]["fields"].append(fid)

            # Build final return structure
            envs_structure = {}
            for eid in active_env_ids:
                envs_structure[eid] = {
                    "modules": modules_struct_by_env[eid]
                }

            return {
                "sessions": {
                    session_id: {
                        "envs": envs_structure
                    }
                }
            }

        except Exception as e:
            print("Err get_full_session_structure", e)



    def update_research_files(self, user_id: str, session_id: str, new_urls: List[str]):
        """
        Update the research_files column for a session by merging new URLs with existing ones.
        """
        try:
            # 1. Fetch existing session to get current files
            session = self.get_session(int(session_id))
            if not session:
                print(f"Session {session_id} not found for research file update")
                return

            current_files_str = session.get("research_files", "[]")
            try:
                current_files = json.loads(current_files_str)
                if not isinstance(current_files, list):
                    current_files = []
            except json.JSONDecodeError:
                current_files = []

            # 2. Merge lists (deduplicate)
            merged_files = list(set(current_files + new_urls))
            
            # 3. Update session
            updates = {
                "research_files": json.dumps(merged_files),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            self.qb.set_item("sessions", updates, keys={"id": int(session_id)})
            print(f"{_SESSION_DEBUG} update_research_files: done, count={len(merged_files)}")
        except Exception as e:
            print(f"{_SESSION_DEBUG} update_research_files: error: {e}")
            import traceback
            traceback.print_exc()


# Instantiate
session_manager = SessionManager()

# -- RELAY HANDLERS --

def handle_get_sessions_modules(payload):
    """
    receive "GET_SESSIONS_MODULES": auth={user_id:str, session_id:str} -> SEND_SESSIONS_MODULES
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    
    if not user_id or not session_id:
        return {"error": "Missing user_id or session_id"}
        
    data = session_manager.get_session_modules(user_id, session_id)
    return {
        "type": "GET_SESSIONS_MODULES",
        "data": data
    }

def handle_get_sessions_envs(payload):
    """
    receive "GET_SESSIONS_ENVS": auth={user_id:str, session_id:str} 
    -> receive sessions_to_envs ... -> return type="GET_SESSIONS_ENVS", data=...
    """
    print("handle_get_sessions_envs")
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    
    if not user_id or not session_id:
        return {"error": "Missing user_id or session_id"}
        
    data = session_manager.get_session_envs(user_id, session_id)
    
    # data from retrieve_env_from_id is {"envs": [...] }
    print("finished handle_get_sessions_envs:", data)
    return {
        "type": "GET_SESSIONS_ENVS",
        "data": data 
    }

def handle_sessions_injections(payload):
    """
    receive "SESSIONS_INJECTIONS": data={}, auth={user_id:str, session_id:str}
    -> receive sessions_to_injections ... -> return: type="", data=injections:list[injection-table rows]
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    
    if not user_id or not session_id:
        return {"error": "Missing user_id or session_id"}
        
    data = session_manager.get_session_injections(user_id, session_id)
    # data is {"injections": [...]}
    
    return {
        "type": "SESSIONS_INJECTIONS",
        "data": data
    }

def handle_link_env_session(payload):
    """
    receive type="LINK_ENV_SESSION", auth={user_id:str, session_id:str, env_id} 
    -> insert session_to_envs-table row with status = active and auth data -> SEND_SESSION_ENVS
    """
    print("handle_link_env_session")
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    env_id = auth.get("env_id") 
    
    if not all([user_id, session_id, env_id]):
        return {"error": "Missing required fields"}
        
    session_manager.link_env_session(user_id, session_id, env_id)
    
    # Return hierarchical structure
    structure = session_manager.get_full_session_structure(user_id, session_id)
    print("handle_link_env_session LIST_SESSIONS_ENVS:")

    pprint.pp(structure)

    return {
        "type": "LIST_SESSIONS_ENVS", 
        "data": structure
    }

def handle_rm_link_env_session(payload):
    """
    receive type="RM_LINK_ENV_SESSION", auth={user_id:str, session_id:str, env_id} 
    -> update session_to_envs-table row with status = deleted -> SEND_SESSION_ENVS
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    env_id = auth.get("env_id")
    
    if not all([user_id, session_id, env_id]):
        return {"error": "Missing required fields"}
        
    session_manager.rm_link_env_session(user_id, session_id, env_id)

        # Return hierarchical structure
    structure = session_manager.get_full_session_structure(user_id, session_id)

    return {
        "type": "LIST_SESSIONS_ENVS", # Using same type? Or ENABLE_SM style? Requests usually imply type match.
        "data": structure
    }


def handle_list_user_sessions(payload):
    """
    receive "LIST_USERS_SESSIONS": auth={user_id:str}
    -> return type="LIST_USERS_SESSIONS", data={"sessions": [...]}
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    
    if not user_id:
        return {"error": "Missing user_id"}
        
    sessions = session_manager.list_user_sessions(user_id)
    return {
        "type": "LIST_USERS_SESSIONS",
        "data": {"sessions": sessions}
    }



"""
Injection Management with BigQuery Integration

Handles injection CRUD operations for energy designer data.
Injection format: {id: str, data: [[times], [energies]], ntype: str}
"""

from typing import Optional, Dict, Any, List
from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from core.qbrain_manager import get_qbrain_table_manager

_INJ_DEBUG = "[InjectionManager]"


class InjectionManager:
    """
    Manages injection data in BigQuery.
    Receives BQCore instance via constructor.
    """

    DATASET_ID = "QBRAIN"
    TABLE = "injections"

    def __init__(self, qb):
        """Initialize InjectionManager with QBrainTableManager instance."""
        try:
            self.qb = qb
            self.INJECTION_TABLE_SCHEMA = {
                "id": "STRING",
                "user_id": "STRING",
                "data": "JSON",
                "created_at": "TIMESTAMP",
                "updated_at": "TIMESTAMP",
            }
            self.table = self.TABLE
            print(f"{_INJ_DEBUG} initialized")
        except Exception as e:
            print(f"{_INJ_DEBUG} __init__ error: {e}")
            import traceback
            traceback.print_exc()
            raise



    def _validate_injection_object(self, inj_object: Dict[str, Any]) -> bool:
        """
        Validate injection object structure.
        """
        try:
            if not isinstance(inj_object, dict):
                print("Error: Injection object must be a dictionary")
                return False
            
            # Check required fields
            required_fields = ["id", "data"]
            for field in required_fields:
                if field not in inj_object:
                    print(f"Error: Missing required field '{field}'")
                    return False
            
            # Validate data structure: should be [[times], [energies]]
            data = inj_object["data"]
            if not isinstance(data, list) or len(data) != 2:
                print("Error: 'data' must be a list of 2 arrays [[times], [energies]]")
                return False
            
            if not isinstance(data[0], list) or not isinstance(data[1], list):
                print("Error: Both time and energy must be arrays")
                return False
            
            if len(data[0]) != len(data[1]):
                print("Error: Time and energy arrays must have the same length")
                return False
            
            return True
            
        except Exception as e:
            print(f"{_INJ_DEBUG} _validate_injection_object: error: {e}")
            return False

    def set_inj(
        self, 
        inj_object: Dict[str, Any], 
        user_id: str
        ) -> bool:
        """
        Upsert injection to BigQuery.
        Dynamically executes schema based packing
        """
        print("set_inj...", inj_object)

        try:
            if isinstance(inj_object, list):
                for item in inj_object:
                    print(f"{_INJ_DEBUG} set_inj: user_id={user_id}, inj_id={item.get('id')}")
                    injection_record = item.copy()
                    injection_record["user_id"] = user_id
                    out = self.qb.set_item(self.table, injection_record, keys={"id": item["id"], "user_id": user_id})
            else:
                inj_object["user_id"] = user_id
                out = self.qb.set_item(self.table, inj_object, keys={"id": inj_object["id"], "user_id": user_id})

            print(f"{_INJ_DEBUG} set_inj: done")
            return True
        except Exception as e:
            print(f"{_INJ_DEBUG} set_inj: error: {e}")
            import traceback
            traceback.print_exc()
            return False



    def del_inj(self, injection_id: str, user_id: str) -> bool:
        try:
            print(f"{_INJ_DEBUG} del_inj: injection_id={injection_id}, user_id={user_id}")
            out = self.qb.del_entry(
                nid=injection_id,
                table=self.table,
                user_id=user_id
            )
            print(f"{_INJ_DEBUG} del_inj: done, success={out}")
            return out
        except Exception as e:
            print(f"{_INJ_DEBUG} del_inj: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_inj_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all injections for a specific user using foreign key relationship.
        """
        try:
            print(f"{_INJ_DEBUG} get_inj_user: user_id={user_id}")
            injections = self.qb.get_users_entries(
                user_id=user_id,
                table=self.table
            )
            if injections:
                for inj in injections:
                    if isinstance(inj.get("data"), str):
                        try:
                            inj["data"] = json.loads(inj["data"])
                        except Exception as e:
                            print(f"{_INJ_DEBUG} get_inj_user: parse data: {e}")
            print(f"{_INJ_DEBUG} get_inj_user: got {len(injections) if injections else 0} injection(s)")
            return injections
        except Exception as e:
            print(f"{_INJ_DEBUG} get_inj_user: error: {e}")
            import traceback
            traceback.print_exc()
            return []


    def get_inj_list(self, inj_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get specific injections by ID list.
        """
        try:
            print(f"{_INJ_DEBUG} get_inj_list: inj_ids count={len(inj_ids) if inj_ids else 0}")
            rows = self.qb.row_from_id(
                nid=inj_ids,
                select="*",
                table=self.table
            )
            if rows:
                for inj in rows:
                    if isinstance(inj.get("data"), str):
                        try:
                            inj["data"] = json.loads(inj["data"])
                        except Exception as e:
                            print(f"{_INJ_DEBUG} get_inj_list: parse data: {e}")
            return rows
        except Exception as e:
            print(f"{_INJ_DEBUG} get_inj_list: error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_injection(self, injection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single injection by ID.
        """
        try:
            print(f"{_INJ_DEBUG} get_injection: injection_id={injection_id}")
            rows = self.qb.row_from_id(
                nid=injection_id,
                select="*",
                table=self.table
            )
            if rows:
                inj = rows[0]
                if isinstance(inj.get("data"), str):
                    try:
                        inj["data"] = json.loads(inj["data"])
                    except Exception as e:
                        print(f"{_INJ_DEBUG} get_injection: parse data: {e}")
                return inj
            return None
        except Exception as e:
            print(f"{_INJ_DEBUG} get_injection: error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def link_inj_env(self, injection_id: str, env_id: str, user_id: str, pos:tuple) -> bool:
        """
        Link an injection to an environment.
        
        Args:
             injection_id: ID of the injection
             env_id: ID of the environment
            
        Returns:
             True if successful, False otherwise
        """
        try:
            payload = {
                "id": f"{env_id}_{user_id}_{injection_id}",
                "env_id": env_id,
                "inj_id": injection_id,
                "user_id": user_id,
                "pos": str(pos),
            }
            # Correcting table name to "envs_to_injections" as typically expected for links.
            print(f"Linked injection {injection_id} to environment {env_id}")
            self.qb.set_item("envs_to_injections", payload, keys={"id": payload["id"]})
            return True
            
        except Exception as e:
            print(f"Error linking injection to environment: {e}")
            return False

    def link_session_injection(self, session_id: str, injection_id: str, user_id: str):
        """Link injection to session."""
        row_id = str(random.randint(1000000000, 9999999999))
        row = {
            "id": row_id,
            "session_id": session_id,
            "injection_id": injection_id, # Schema matches? session manager used "injection_id"
            "user_id": user_id,
        }
        # Note: SessionManager had SESSIONS_TO_INJECTIONS_SCHEMA "session_to_injections"
        self.qb.set_item("session_to_injections", row, keys={"id": row["id"]})


    def rm_link_inj_env(
            self,
            injection_id: str,
            env_id: str,
            user_id: str,
    ) -> bool:
        """
        Remove link between an injection and an environment.
        
        Args:
            injection_id: ID of the injection
            env_id: ID of the environment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.qb.rm_env_link(
                env_id=env_id,
                injection_id=injection_id,
                user_id=user_id,
                table_name="envs_to_injections",
                linked_row_id="inj_id",
                linked_row_id_name="injection_id"
            )
            return True
        except Exception as e:
            print(f"{_INJ_DEBUG} rm_link_inj_env: error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_inj_env(self, env_id: str, user_id: str, injection_id: str, select: str = "*") -> List[Dict[str, Any]]:
        """
        Get all injections linked to a specific environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            List of injection records
        """
        try:
            injections = self.qb.get_envs_linked_rows(
                env_id=env_id,
                user_id=user_id,
                table_name="envs_to_injections",
                linked_row_id=injection_id,
                linked_row_id_name="injection_id",
                select="*"
            )
            
            print(f"{_INJ_DEBUG} get_inj_env: got {len(injections)} injection(s) for env_id={env_id}")
            return injections
        except Exception as e:
            print(f"{_INJ_DEBUG} get_inj_env: error: {e}")
            import traceback
            traceback.print_exc()
            return []


    def rm_link_session_injection(self, session_id: str, injection_id: str, user_id: str):
        """Remove link injection to session (soft delete)."""
        self.qb.rm_link_session_link(
            session_id=session_id,
            nid=injection_id,
            user_id=user_id,
            session_link_table="session_to_injections",
            session_to_link_name_id="injection_id",
        )

    def retrieve_session_injections(self, session_id: str, user_id: str) -> List[Dict[str, Any]]:
        # get session_to_injections rows -> get injection rows

        try:
            links = self.qb.list_session_entries(
                user_id=user_id,
                session_id=session_id,
                table=f"session_to_injections",
                select="injection_id"
            )

            if not links:
                return []

            inj_ids = [l["injection_id"] for l in links]

            injections = self.qb.row_from_id(
                nid=inj_ids,
                select="*",
                table=self.table, # injections table
                user_id=user_id
            )
            if injections:
                for inj in injections:
                    if isinstance(inj.get("data"), str):
                        try:
                            inj["data"] = json.loads(inj["data"])
                        except Exception as e:
                            print(f"{_INJ_DEBUG} retrieve_session_injections: parse data: {e}")
                print(f"{_INJ_DEBUG} retrieve_session_injections: got {len(injections) if injections else 0} injection(s)")
                return injections
        except Exception as e:
            print(f"{_INJ_DEBUG} retrieve_session_injections: error: {e}")
            import traceback
            traceback.print_exc()
            return []


_default_bqcore = BQCore(dataset_id="QBRAIN")
_default_injection_manager = InjectionManager(get_qbrain_table_manager(_default_bqcore))
injection_manager = _default_injection_manager  # backward compat

import random

# ============================================================================
# RELAY HANDLERS
# ============================================================================

from core.websocket_datatypes import (
    WebSocketRequest, WebSocketResponse, AuthData,
    create_list_response
)

import json

from core.handler_utils import require_param, require_param_truthy, get_val


def handle_get_injection(data=None, auth=None) -> dict:
    """Retrieve a single injection by ID. Required: injection_id or id (auth or data)."""
    from core.managers_context import get_injection_manager
    data, auth = data or {}, auth or {}
    injection_id = get_val(data, auth, "injection_id") or get_val(data, auth, "id")
    if err := require_param(injection_id, "injection_id"):
        return err
    auth_out = {k: v for k, v in {**auth, **data}.items() if k in ("user_id", "injection_id") and v is not None}
    try:
        injection = get_injection_manager().get_injection(injection_id)
        if not injection:
            return WebSocketResponse.error(type="get_injection", error="Injection not found", auth=auth_out)
        return WebSocketResponse.success(type="GET_INJECTION", data=injection, auth=auth_out)
    except Exception as e:
        return WebSocketResponse.error(type="get_injection", error=str(e), auth=auth_out)


def handle_set_inj(data=None, auth=None) -> dict:
    """Create or update an injection. Required: user_id (auth), data (injection dict with id/data/ntype). Optional: original_id (auth)."""
    user_id = get_val(data, auth, "user_id")
    inj_dict = data if isinstance(data, dict) else None
    original_id = get_val(data, auth, "original_id")

    if err := require_param(user_id, "user_id"):
        return err

    if err := require_param_truthy(inj_dict, "data"):
        return err

    auth_out = {"user_id": user_id}

    from core.managers_context import get_injection_manager
    im = get_injection_manager()
    try:
        if original_id:
            im.del_inj(original_id, user_id)
        success = im.set_inj(data, user_id)

        if not success:
            print("!success", success)
            return WebSocketResponse.error("SET_INJ", "Failed to save injection", auth=auth_out)
        return {"type": "GET_INJ_USER", "data": {"injections": im.get_inj_user(user_id)}}
    except Exception as e:
        print("Err setting injection:", e)
        return WebSocketResponse.error("SET_INJ", str(e), auth=auth_out)


def handle_del_inj(data=None, auth=None) -> dict:
    """Delete an injection by ID. Required: user_id, injection_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    injection_id = get_val(data, auth, "injection_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(injection_id, "injection_id"):
        return err
    from core.managers_context import get_injection_manager
    im = get_injection_manager()
    auth_out = {"user_id": user_id}
    try:
        success = im.del_inj(injection_id, user_id)
        if not success:
            return WebSocketResponse.error("DEL_INJ", "Failed to delete injection", auth=auth_out)
        return {"type": "GET_INJ_USER", "data": {"injections": im.get_inj_user(user_id)}}
    except Exception as e:
        return WebSocketResponse.error("DEL_INJ", str(e), auth=auth_out)


def handle_get_inj_user(data=None, auth=None) -> dict:
    """Retrieve all injections owned by a user. Required: user_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    if err := require_param(user_id, "user_id"):
        return err
    from core.managers_context import get_injection_manager
    try:
        return {"type": "GET_INJ_USER", "data": {"injections": get_injection_manager().get_inj_user(user_id)}}
    except Exception as e:
        print("Err handle_get_inj_user:", e)
        return {"type": "GET_INJ_USER", "data": {"injections": []}}


def handle_get_inj_list(data=None, auth=None) -> dict:
    """Retrieve injections by a list of IDs. Required: user_id (auth or data), inj_ids (data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    inj_ids = data.get("inj_ids") if isinstance(data, dict) else []
    if err := require_param(user_id, "user_id"):
        return err
    if not isinstance(inj_ids, list):
        return {"error": "param missing", "key": "inj_ids"}
    auth_out = {"user_id": user_id}
    from core.managers_context import get_injection_manager
    try:
        return create_list_response("GET_INJ_LIST", get_injection_manager().get_inj_list(inj_ids), auth=auth_out)
    except Exception as e:
        return create_list_response("GET_INJ_LIST", [], error=str(e), auth=auth_out)


def handle_link_inj_env(data=None, auth=None) -> dict:
    """Link an injection to an environment. Required: injection_id, env_id, user_id (auth or data)."""
    data, auth = data or {}, auth or {}
    injection_id = get_val(data, auth, "injection_id")
    env_id = get_val(data, auth, "env_id")
    user_id = get_val(data, auth, "user_id")
    if err := require_param(injection_id, "injection_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    if err := require_param(user_id, "user_id"):
        return err
    from core.managers_context import get_injection_manager
    im = get_injection_manager()
    auth_out = {"user_id": user_id, "env_id": env_id}
    try:
        im.link_inj_env(injection_id, env_id, user_id, pos=(0, 0, 0))
        return create_list_response("GET_INJ_ENV", im.get_inj_env(env_id, user_id, injection_id), auth=auth_out)
    except Exception as e:
        return WebSocketResponse.error("LINK_INJ_ENV", str(e), auth=auth_out)


def handle_rm_link_inj_env(data=None, auth=None) -> dict:
    """Remove the link between an injection and an environment. Required: user_id, env_id, injection_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    env_id = get_val(data, auth, "env_id")
    injection_id = get_val(data, auth, "injection_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    if err := require_param(injection_id, "injection_id"):
        return err
    from core.managers_context import get_injection_manager
    im = get_injection_manager()
    auth_out = {"user_id": user_id}
    try:
        im.rm_link_inj_env(injection_id, env_id, user_id)
        return create_list_response("GET_INJ_ENV", im.get_inj_env(env_id, user_id, injection_id), auth=auth_out)
    except Exception as e:
        return WebSocketResponse.error("RM_LINK_INJ_ENV", str(e), auth=auth_out)


def handle_list_link_inj_env(data=None, auth=None) -> dict:
    """List injections linked to an environment. Required: env_id, user_id (auth or data)."""
    data, auth = data or {}, auth or {}
    env_id = get_val(data, auth, "env_id")
    user_id = get_val(data, auth, "user_id")
    injection_id = get_val(data, auth, "injection_id")
    if err := require_param(env_id, "env_id"):
        return err
    if err := require_param(user_id, "user_id"):
        return err
    from core.managers_context import get_injection_manager
    auth_out = {"user_id": user_id}
    try:
        return create_list_response("GET_INJ_ENV", get_injection_manager().get_inj_env(env_id, user_id, injection_id), auth=auth_out)
    except Exception as e:
        return WebSocketResponse.error("LIST_LINK_INJ_ENV", str(e), auth=auth_out)


def handle_get_sessions_injections(data=None, auth=None):
    """Retrieve all injections linked to a session. Required: user_id, session_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    from core.managers_context import get_injection_manager
    return {"type": "GET_SESSIONS_INJECTIONS", "data": {"injections": get_injection_manager().retrieve_session_injections(session_id, user_id)}}


def handle_link_session_injection(data=None, auth=None):
    """Link a session to an injection. Required: user_id, session_id, injection_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    injection_id = get_val(data, auth, "injection_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    if err := require_param(injection_id, "injection_id"):
        return err
    from core.managers_context import get_injection_manager
    get_injection_manager().link_session_injection(session_id, injection_id, user_id)
    return handle_get_sessions_injections(data={"session_id": session_id}, auth={"user_id": user_id})


def handle_rm_link_session_injection(data=None, auth=None):
    """Remove the link between a session and an injection. Required: user_id, session_id, injection_id (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    session_id = get_val(data, auth, "session_id")
    injection_id = get_val(data, auth, "injection_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(session_id, "session_id"):
        return err
    if err := require_param(injection_id, "injection_id"):
        return err
    from core.managers_context import get_injection_manager
    get_injection_manager().rm_link_session_injection(session_id, injection_id, user_id)
    return handle_get_sessions_injections(data={"session_id": session_id}, auth={"user_id": user_id})

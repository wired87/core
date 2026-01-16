"""
Injection Management with BigQuery Integration

Handles injection CRUD operations for energy designer data.
Injection format: {id: str, data: [[times], [energies]], ntype: str}
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from core.qbrain_manager import QBrainTableManager

class InjectionManager(BQCore):
    """
    Manages injection data in BigQuery.
    Extends BQCore to leverage existing BigQuery functionality.
    """

    DATASET_ID = "QBRAIN"
    
    def __init__(self):
        """Initialize InjectionManager with QBRAIN dataset."""
        BQCore.__init__(self, dataset_id=self.DATASET_ID)
        # Hardcoded schema as per user request
        self.INJECTION_TABLE_SCHEMA = {
            "id": "STRING",
            "user_id": "STRING",
            "data": "JSON",
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        }
        self.qb = QBrainTableManager()
        self.table= f"injections"



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
            print(f"Error validating injection object: {e}")
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

        try:
            
            # Prepare record
            # We want to use the input object, but sanitize/fill it.
            injection_record = inj_object.copy()
            
            # Iterate over schema to map fields
            injection_record["user_id"] = user_id
            
            if "ntype"  in injection_record:
                injection_record.pop("ntype")

            print(f"Inserting injection: {inj_object.get('id')} - {injection_record.get('data')} for user {user_id}")
            
            # Use set_item
            return self.qb.set_item(self.table, injection_record, keys={"id": injection_record["id"], "user_id": user_id})
            
        except Exception as e:
            print(f"Error setting injection: {e}")
            import traceback
            traceback.print_exc()
            return False



    def del_inj(self, injection_id: str, user_id: str) -> bool:
         print(f"del_inj, {injection_id}")
         return self.qb.del_entry(
            nid=injection_id,
            table=self.table,
            user_id=user_id
         )

    def get_inj_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all injections for a specific user using foreign key relationship.
        """
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
                        print(f"Error parsing injection data: {e}")
                        pass
        return injections


    def get_inj_list(self, inj_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get specific injections by ID list.
        """
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
                        print(f"Error parsing injection data: {e}")
                        pass
        return rows

    def get_injection(self, injection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single injection by ID.
        """
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
                    print(f"Error getting injection: {e}")
            return inj
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
            print(f"Error removing link for injection: {e}")
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
            
            print(f"Retrieved {len(injections)} injection(s) for environment {env_id}")
            return injections
            
        except Exception as e:
            print(f"Error retrieving environment injections: {e}")
            return []


    def rm_link_session_injection(self, session_id: str, injection_id: str, user_id: str):
        """Remove link injection to session (soft delete)."""
        self.qb.rm_link_session_link(
            session_id=session_id,
            nid=injection_id,
            user_id=user_id,
            session_link_table=self.table,
            session_to_link_name_id="injection_id",
        )

    def retrieve_session_injections(self, session_id: str, user_id: str) -> List[Dict[str, Any]]:
        # get session_to_injections rows -> get injection rows

        
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
                        print(f"Error parsing injection data: {e}")
                        pass
        return injections


injection_manager = InjectionManager()

import random

# ============================================================================
# RELAY HANDLERS
# ============================================================================

from core.websocket_datatypes import (
    WebSocketRequest, WebSocketResponse, AuthData, InjectionData,
    create_list_response
)
import json

def handle_get_injection(payload: dict) -> dict:
    """
    Handle get_injection WebSocket message.
    Return a single injection by ID.
    """
    auth: AuthData = payload["auth"]
    try:
        # Request parsing
        request = WebSocketRequest.from_dict(payload)
        injection_id = request.get_data_field("id") or request.get_data_field("injection_id")
        
        # Auth usage
        # We assume auth is passed from Relay (enriched with connection user_id)
        
        if not injection_id:
            return WebSocketResponse.error(
                type="get_injection",
                error="Missing injection id",
                auth=auth
            )
            
        injection = injection_manager.get_injection(injection_id)
        
        if not injection:
            return WebSocketResponse.error(
                type="get_injection",
                error="Injection not found",
                auth=auth
            )
            
        return WebSocketResponse.success(
            type="GET_INJECTION",
            data=injection,
            auth=auth
        )
        
    except Exception as e:
        return WebSocketResponse.error(
            type="get_injection",
            error=str(e),
            auth=auth
        )

def handle_set_inj(payload: dict) -> dict:
    """Set/upsert injection."""
    auth = payload["auth"]
    try:
        user_id = auth["user_id"]
        inj_dict = payload.get("data")

        original_id = auth.get("original_id")
        if original_id:
             injection_manager.del_inj(original_id, user_id)

        success = injection_manager.set_inj(inj_dict, user_id)
        if not success:
             return WebSocketResponse.error(
                "set_inj", 
                "Failed to save injection", 
                auth=auth
            )
             
        # Return updated list
        injections = injection_manager.get_inj_user(user_id)
        return {"type":"GET_INJ_USER", "data":{"injections":injections}}

    except Exception as e:
        print("Err setting injection:", e)
        return WebSocketResponse.error("SET_INJ", str(e), auth=auth)


def handle_del_inj(payload: dict) -> dict:
    """Delete injection."""
    auth:dict = payload["auth"]
    try:
        user_id = auth["user_id"]
        injection_id = auth["injection_id"]

        if not injection_id:
             return WebSocketResponse.error("del_inj", "Missing injection id", auth=auth)

        success = injection_manager.del_inj(injection_id, user_id)
        if not success:
             return WebSocketResponse.error("DEL_INJ", "Failed to delete injection", auth=auth)
             
        injections = injection_manager.get_inj_user(user_id)
        return {"type":"GET_INJ_USER", "data":{"injections":injections}}
        
    except Exception as e:
        return WebSocketResponse.error("DEL_INJ", str(e), auth=auth)

def handle_get_inj_user(payload: dict) -> dict:
    """Get all injections for user."""
    auth: dict = payload["auth"]
    try:
        # request = WebSocketRequest.from_dict(payload) # not strictly needed if we trust auth
        user_id = auth["user_id"]
        injections = injection_manager.get_inj_user(user_id)
        return {"type":"GET_INJ_USER", "data":{"injections":injections}}
    except Exception as e:
        print("Err handle_get_inj_user:", e)
        return {"type":"GET_INJ_USER", "data":{"injections": []}}

def handle_get_inj_list(payload: dict) -> dict:
    """Get list of injections by ID."""
    auth: AuthData = payload["auth"]

    try:
        request = WebSocketRequest.from_dict(payload)
        inj_ids = request.get_data_field("inj_ids", [])
        if not isinstance(inj_ids, list):
             return create_list_response("get_inj_list", [], error="inj_ids must be a list", auth=auth)
        
        injections = injection_manager.get_inj_list(inj_ids)
        return create_list_response("GET_INJ_LIST", injections, auth=auth)
    except Exception as e:
        return create_list_response("GET_INJ_LIST", [], error=str(e), auth=auth)

def handle_link_inj_env(payload: dict) -> dict:
    """Link injection to env."""
    auth = payload["auth"]
    try:
        injection_id = auth["injection_id"]
        env_id = auth["env_id"]
        user_id = auth["user_id"]
        if not injection_id or not env_id:
             return WebSocketResponse.error("link_inj_env", "Missing injection_id or env_id", auth=auth)
             
        injection_manager.link_inj_env(injection_id, env_id, user_id, pos=(0,0,0))
        # Return updated env injections
        injections = injection_manager.get_inj_env(env_id, user_id, injection_id)
        return create_list_response("GET_INJ_ENV", injections, auth=auth)
    except Exception as e:
        return WebSocketResponse.error("LINK_INJ_ENV", str(e), auth=auth)

def handle_rm_link_inj_env(payload: dict) -> dict:
    """Remove link."""
    auth: dict = payload["auth"]
    try:
        user_id = auth["user_id"]
        env_id = auth["env_id"]
        injection_id = auth["injection_id"]

        injection_manager.rm_link_inj_env(injection_id, env_id, user_id)
        injections = injection_manager.get_inj_env(env_id, user_id, injection_id)
        return create_list_response("GET_INJ_ENV", injections, auth=auth)
    except Exception as e:
         return WebSocketResponse.error("RM_LINK_INJ_ENV", str(e), auth=auth)

def handle_list_link_inj_env(payload: dict) -> dict:
    """List linked injections."""
    auth = payload["auth"]

    try:
        env_id = auth["env_id"]
        user_id = auth["user_id"]
        injection_id = auth["injection_id"]

        if not env_id:
             return WebSocketResponse.error("LIST_LINK_INJ_ENV", "Missing creds", auth=auth)
        injections = injection_manager.get_inj_env(env_id, user_id, injection_id)
        return create_list_response("GET_INJ_ENV", injections, auth=auth)
    except Exception as e:
        return WebSocketResponse.error("LIST_LINK_INJ_ENV", str(e), auth=auth)
        return WebSocketResponse.error("list_link_inj_env", str(e), auth=auth)

def handle_get_sessions_injections(payload):
    """
    receive "GET_SESSIONS_INJECTIONS"
    """
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    if not user_id or not session_id:
        return {"error": "Missing user_id or session_id"}
        
    injections = injection_manager.retrieve_session_injections(session_id, user_id)
    return {
        "type": "GET_SESSIONS_INJECTIONS",
        "data": {"injections": injections}
    }

def handle_link_session_injection(payload):
    """Link session to injection."""
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    injection_id = auth.get("injection_id")
    
    if not all([user_id, session_id, injection_id]):
        return {"error": "Missing required fields"}
        
    injection_manager.link_session_injection(session_id, injection_id, user_id)
    return handle_get_sessions_injections(payload)

def handle_rm_link_session_injection(payload):
    """Remove link."""
    auth = payload.get("auth", {})
    user_id = auth.get("user_id")
    session_id = auth.get("session_id")
    injection_id = auth.get("injection_id")
    
    if not all([user_id, session_id, injection_id]):
        return {"error": "Missing required fields"}
        
    injection_manager.rm_link_session_injection(session_id, injection_id, user_id)
    return handle_get_sessions_injections(payload)

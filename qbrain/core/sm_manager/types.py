"""
Type definitions for SMManager handler methods.
Captures req_struct and out_struct data types for each relay case.
"""
from typing import TypedDict, Dict, Any


# =============================================================================
# DATATYPES - Exact data structures used by handler methods (nested list[dict])
# =============================================================================

# ENABLE_SM has no data key; all from auth (env_id, session_id, user_id)
# No ReqData type needed for ENABLE_SM


# --- Auth types ---
class AuthEnvSessionUserId(TypedDict):
    env_id: str
    session_id: str
    user_id: str


# --- Request payload data types ---
class ReqEnableSm(TypedDict):
    auth: AuthEnvSessionUserId


# --- Response data types ---
class OutEnableSm(TypedDict):
    type: str  # "ENABLE_SM"
    data: Dict[str, Any]  # {"sessions": {...}}


# --- Relay case struct type ---
class RelayCaseStruct(TypedDict, total=False):
    case: str
    desc: str
    func: Any
    func_name: None
    req_struct: Dict[str, Any]
    out_struct: Dict[str, Any]

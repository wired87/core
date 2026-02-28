"""
Type definitions for ModelManager handler methods.
Captures req_struct and out_struct data types for each relay case.
"""
from typing import TypedDict, Dict, Any


# =============================================================================
# DATATYPES - Exact data structures used by handler methods (nested list[dict])
# =============================================================================

class ReqDataQueryModel(TypedDict):
    """req_struct.data for QUERY_MODEL. Handler passes data['question'] to query()."""
    question: str


# --- Auth types ---
class AuthUserEnvId(TypedDict):
    user_id: str
    env_id: str


# --- Request payload data types ---
class ReqQueryModelData(TypedDict):
    question: str


class ReqQueryModel(TypedDict):
    auth: AuthUserEnvId
    data: ReqQueryModelData


# --- Response data types ---
class OutQueryModel(TypedDict):
    type: str  # "QUERY_MODEL_RESPONSE"
    data: Dict[str, str]  # {"answer": str, "env_id": str}


# --- Relay case struct type ---
class RelayCaseStruct(TypedDict, total=False):
    case: str
    desc: str
    func: Any
    func_name: None
    req_struct: Dict[str, Any]
    out_struct: Dict[str, Any]

"""
Type definitions for EnvManager handler methods.
Captures req_struct and out_struct data types for each relay case.
"""
from typing import TypedDict, List, Dict, Any, Optional


# =============================================================================
# DATATYPES - Exact data structures used by handler methods (nested list[dict])
# =============================================================================

class EnvItemData(TypedDict, total=False):
    """Exact type for env passed to EnvManager.set_env(). From ENV_SCHEMA."""
    id: str
    sim_time: int
    cluster_dim: int
    dims: int
    user_id: str
    data: Optional[str]  # JSON string when persisted


class ReqDataSetEnv(TypedDict):
    """req_struct.data for SET_ENV. Handler passes data['env'] to set_env()."""
    env: EnvItemData


# --- Auth types ---
class AuthEnvId(TypedDict):
    env_id: str


class AuthUserId(TypedDict):
    user_id: str


class AuthEnvUserId(TypedDict):
    env_id: str
    user_id: str


class AuthUserOriginalId(TypedDict):
    user_id: str
    original_id: Optional[str]


class AuthEnvData(TypedDict):
    env: Dict[str, Any]


# --- Request payload data types ---
class ReqGetEnv(TypedDict):
    auth: AuthEnvId


class ReqGetUsersEnvs(TypedDict):
    auth: AuthUserId


class ReqDelEnv(TypedDict):
    auth: AuthEnvUserId


class ReqSetEnv(TypedDict):
    data: AuthEnvData
    auth: AuthUserOriginalId


class ReqDownloadModel(TypedDict):
    auth: AuthEnvUserId


class ReqRetrieveLogsEnv(TypedDict):
    auth: AuthEnvUserId


class ReqGetEnvData(TypedDict):
    auth: AuthEnvUserId


# --- Response data types ---
class OutGetEnv(TypedDict):
    type: str  # "GET_ENV"
    data: Dict[str, Any]  # {"env": ...} or {"envs": ...}


class OutGetUsersEnvs(TypedDict):
    type: str  # "GET_USERS_ENVS"
    data: Dict[str, List[Dict[str, Any]]]  # {"envs": [...]}


class OutDownloadModel(TypedDict):
    type: str  # "DOWNLOAD_MODEL"
    data: Dict[str, Any]


class OutRetrieveLogsEnv(TypedDict):
    type: str  # "RETRIEVE_LOGS_ENV"
    data: List[Any]


class OutGetEnvData(TypedDict):
    type: str  # "GET_ENV_DATA"
    data: List[Any]


# --- Relay case struct type ---
class RelayCaseStruct(TypedDict, total=False):
    case: str
    desc: str
    func: Any
    func_name: Optional[str]
    req_struct: Dict[str, Any]
    out_struct: Dict[str, Any]

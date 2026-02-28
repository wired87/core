"""
Type definitions for SessionManager handler methods.
Captures req_struct and out_struct data types for each relay case.
"""
from typing import TypedDict, List, Dict, Any, Optional


# --- Auth types ---
class AuthUserSessionEnvId(TypedDict):
    user_id: str
    session_id: str
    env_id: str


class AuthUserSessionId(TypedDict):
    user_id: str
    session_id: str


class AuthUserId(TypedDict):
    user_id: str


# --- Request payload data types ---
class ReqLinkEnvSession(TypedDict):
    auth: AuthUserSessionEnvId


class ReqRmLinkEnvSession(TypedDict):
    auth: AuthUserSessionEnvId


class ReqGetSessionsEnvs(TypedDict):
    auth: AuthUserSessionId


class ReqListUsersSessions(TypedDict):
    auth: AuthUserId


# --- Response data types ---
class OutLinkEnvSession(TypedDict):
    type: str  # "LINK_ENV_SESSION"
    data: Dict[str, Any]  # {"sessions": {...}}


class OutGetSessionsEnvs(TypedDict):
    type: str  # "GET_SESSIONS_ENVS"
    data: Dict[str, List[Dict[str, Any]]]  # {"envs": [...]}


class OutListUsersSessions(TypedDict):
    type: str  # "LIST_USERS_SESSIONS"
    data: Dict[str, List[Dict[str, Any]]]  # {"sessions": [...]}


# --- Relay case struct type ---
class RelayCaseStruct(TypedDict, total=False):
    case: str
    desc: str
    func: Any
    func_name: Optional[str]
    req_struct: Dict[str, Any]
    out_struct: Dict[str, Any]

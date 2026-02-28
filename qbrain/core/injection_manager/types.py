"""
Type definitions for InjectionManager handler methods.
Captures req_struct and out_struct data types for each relay case.
"""
from typing import TypedDict, List, Dict, Any, Optional, Union


# =============================================================================
# DATATYPES - Exact data structures used by handler methods (nested list[dict])
# =============================================================================

# Injection format: {id: str, data: [[times], [energies]], ntype: str}
# data is list of 2 lists: [[times], [energies]] - List[List[float]]
class InjectionDataTimesEnergies(TypedDict, total=False):
    """Nested structure: data[0]=times, data[1]=energies."""
    pass  # Represented as List[List[float]]


class InjectionItemData(TypedDict, total=False):
    """Exact type for injection passed to InjectionManager.set_inj()."""
    id: str
    data: List[List[float]]  # [[times], [energies]]
    ntype: Optional[str]


class ReqDataSetInj(TypedDict):
    """req_struct.data for SET_INJ. Handler passes payload['data'] to set_inj()."""
    pass  # data is the injection dict itself (InjectionItemData)


class ReqDataGetInjList(TypedDict):
    """req_struct.data for GET_INJ_LIST."""
    inj_ids: List[str]


class ReqDataGetInjection(TypedDict, total=False):
    """req_struct.data for GET_INJECTION. id or injection_id."""
    id: Optional[str]
    injection_id: Optional[str]


# --- Auth types ---
class AuthUserId(TypedDict):
    user_id: str


class AuthUserOriginalId(TypedDict):
    user_id: str
    original_id: Optional[str]


class AuthInjectionUserId(TypedDict):
    injection_id: str
    user_id: str


class AuthInjEnvUserId(TypedDict):
    injection_id: str
    env_id: str
    user_id: str


class AuthEnvUserId(TypedDict):
    env_id: str
    user_id: str


class AuthSessionInjectionUserId(TypedDict):
    session_id: str
    injection_id: str
    user_id: str


class AuthSessionUserId(TypedDict):
    session_id: str
    user_id: str


# --- Request payload data types ---
class ReqSetInj(TypedDict):
    data: Dict[str, Any]
    auth: AuthUserOriginalId


class ReqDelInj(TypedDict):
    auth: AuthInjectionUserId


class ReqGetInjUser(TypedDict):
    auth: AuthUserId


class ReqGetInjList(TypedDict):
    data: Dict[str, List[str]]  # {"inj_ids": [...]}
    auth: AuthUserId


class ReqLinkInjEnv(TypedDict):
    auth: AuthInjEnvUserId


class ReqListLinkInjEnv(TypedDict):
    auth: AuthEnvUserId


class ReqGetInjection(TypedDict):
    auth: Dict[str, str]  # injection_id or id in data
    data: Optional[Dict[str, str]]  # id, injection_id


class ReqLinkSessionInjection(TypedDict):
    auth: AuthSessionInjectionUserId


class ReqGetSessionsInjections(TypedDict):
    auth: AuthSessionUserId


# --- Response data types ---
class OutGetInjUser(TypedDict):
    type: str  # "GET_INJ_USER"
    data: Dict[str, List[Dict[str, Any]]]  # {"injections": [...]}


class OutGetInjList(TypedDict):
    type: str  # "GET_INJ_LIST"
    data: List[Dict[str, Any]]


class OutGetInjEnv(TypedDict):
    type: str  # "GET_INJ_ENV"
    data: List[Dict[str, Any]]


class OutGetInjection(TypedDict):
    type: str  # "GET_INJECTION"
    data: Dict[str, Any]


class OutGetSessionsInjections(TypedDict):
    type: str  # "GET_SESSIONS_INJECTIONS"
    data: Dict[str, List[Dict[str, Any]]]  # {"injections": [...]}


# --- Relay case struct type ---
class RelayCaseStruct(TypedDict, total=False):
    case: str
    desc: str
    func: Any
    func_name: Optional[str]
    req_struct: Dict[str, Any]
    out_struct: Optional[Dict[str, Any]]

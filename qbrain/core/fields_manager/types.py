"""
Type definitions for FieldsManager handler methods.
Captures req_struct and out_struct data types for each relay case.
"""
from typing import TypedDict, List, Dict, Any, Optional


# =============================================================================
# DATATYPES - Exact data structures used by handler methods (nested list[dict])
# =============================================================================

class FieldItemData(TypedDict, total=False):
    """Exact type for field passed to FieldsManager.set_field(). From FIELD_SCHEMA."""
    id: str
    params: str  # JSON string of param IDs
    module: Optional[str]
    linked_fields: Optional[List[str]]  # Renamed to interactant_fields in handler
    interactant_fields: Optional[List[str]]


class ReqDataSetField(TypedDict):
    """req_struct.data for SET_FIELD. Handler passes data['field'] to set_field()."""
    field: FieldItemData


class LinkModuleFieldItem(TypedDict, total=False):
    """Single link item for link_module_field. Handler builds list from qbrain.auth."""
    id: str
    module_id: str
    field_id: str
    session_id: Optional[str]
    env_id: Optional[str]
    user_id: str
    status: str


class ReqDataLinkModuleField(TypedDict):
    """Handler builds List[LinkModuleFieldItem] from auth (user_id, module_id, field_id, session_id, env_id)."""
    pass  # No data key; all from auth


# --- Auth types ---
class AuthFieldUserId(TypedDict):
    field_id: str
    user_id: str


class AuthUserId(TypedDict):
    user_id: str


class AuthModuleUserId(TypedDict):
    module_id: str
    user_id: str


class AuthUserOriginalId(TypedDict):
    user_id: str
    original_id: Optional[str]


class AuthUserModuleFieldId(TypedDict):
    user_id: str
    module_id: str
    field_id: str


class AuthUserSessionId(TypedDict):
    user_id: str
    session_id: str


# --- Request payload data types ---
class ReqSetFieldData(TypedDict):
    field: Dict[str, Any]


class ReqDelField(TypedDict):
    auth: AuthFieldUserId


class ReqListUsersFields(TypedDict):
    auth: AuthUserId


class ReqListModulesFields(TypedDict):
    auth: AuthModuleUserId


class ReqSetField(TypedDict):
    data: ReqSetFieldData
    auth: AuthUserOriginalId


class ReqLinkModuleField(TypedDict):
    auth: AuthUserModuleFieldId


class ReqRmLinkModuleField(TypedDict):
    auth: AuthUserModuleFieldId


class ReqGetModulesFields(TypedDict):
    auth: AuthModuleUserId


class ReqGetSessionsFields(TypedDict):
    auth: AuthUserSessionId


# --- Response data types ---
class OutListUsersFields(TypedDict):
    type: str  # "LIST_USERS_FIELDS"
    data: Dict[str, List[Dict[str, Any]]]  # {"fields": [...]}


class OutGetModulesFields(TypedDict):
    type: str  # "GET_MODULES_FIELDS"
    data: Dict[str, List[Dict[str, Any]]]  # {"fields": [...]}


class OutSessionsFields(TypedDict):
    type: str  # "SESSIONS_FIELDS"
    data: Dict[str, List[Dict[str, Any]]]  # {"fields": [...]}


# --- Relay case struct type ---
class RelayCaseStruct(TypedDict, total=False):
    case: str
    desc: str
    func: Any
    func_name: Optional[str]
    req_struct: Dict[str, Any]
    out_struct: Dict[str, Any]

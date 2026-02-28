from .fields_lib import (
    handle_del_field, handle_set_field, handle_link_module_field,
    handle_list_modules_fields, handle_list_users_fields,
    handle_rm_link_module_field, handle_get_modules_fields,
    handle_get_sessions_fields
)
from .types import (
    FieldItemData, ReqDataSetField, LinkModuleFieldItem,
    OutListUsersFields, OutGetModulesFields, OutSessionsFields, RelayCaseStruct,
)

# Case structs - req_struct.data uses exact datatypes (FieldItemData, ReqDataSetField)
DEL_FIELD_CASE: RelayCaseStruct = {
    "case": "DEL_FIELD", "desc": "Delete Field", "func": handle_del_field,
    "req_struct": {"auth": {"field_id": str, "user_id": str}},
    "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": list}},  # OutListUsersFields
}
LIST_USERS_FIELDS_CASE: RelayCaseStruct = {
    "case": "LIST_USERS_FIELDS", "desc": "List Users Fields", "func": handle_list_users_fields,
    "req_struct": {"auth": {"user_id": str}},
    "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": list}},  # OutListUsersFields
}
LIST_MODULES_FIELDS_CASE: RelayCaseStruct = {
    "case": "LIST_MODULES_FIELDS", "desc": "List Modules Fields", "func": handle_list_modules_fields,
    "req_struct": {"auth": {"module_id": str, "user_id": str}},
    "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": list}},  # OutGetModulesFields
}
SET_FIELD_CASE: RelayCaseStruct = {
    "case": "SET_FIELD", "desc": "Set Field", "func": handle_set_field,
    "req_struct": {
        "data": {"field": {"id": str, "params": str, "module": str, "linked_fields": list}},  # ReqDataSetField
        "auth": {"user_id": str, "original_id": str}
    },
    "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": list}},  # OutListUsersFields
}
LINK_MODULE_FIELD_CASE: RelayCaseStruct = {
    "case": "LINK_MODULE_FIELD", "desc": "Link Module Field", "func": handle_link_module_field,
    "req_struct": {"auth": {"user_id": str, "module_id": str, "field_id": str, "session_id": str, "env_id": str}},  # ReqDataLinkModuleField (all from auth)
    "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": list}},  # OutGetModulesFields
}
RM_LINK_MODULE_FIELD_CASE: RelayCaseStruct = {
    "case": "RM_LINK_MODULE_FIELD", "desc": "Remove Link Module Field", "func": handle_rm_link_module_field,
    "req_struct": {"auth": {"user_id": str, "module_id": str, "field_id": str}},
    "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": list}},  # OutGetModulesFields
}
GET_MODULES_FIELDS_CASE: RelayCaseStruct = {
    "case": "GET_MODULES_FIELDS", "desc": "Get Modules Fields", "func": handle_get_modules_fields,
    "req_struct": {"auth": {"user_id": str, "module_id": str}},
    "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": list}},  # OutGetModulesFields
}
SESSIONS_FIELDS_CASE: RelayCaseStruct = {
    "case": "SESSIONS_FIELDS", "desc": "Get Sessions Fields", "func": handle_get_sessions_fields,
    "req_struct": {"auth": {"user_id": str, "session_id": str}},
    "out_struct": {"type": "SESSIONS_FIELDS", "data": {"fields": list}},  # OutSessionsFields
}

RELAY_FIELD = [
    DEL_FIELD_CASE, LIST_USERS_FIELDS_CASE, LIST_MODULES_FIELDS_CASE, SET_FIELD_CASE,
    LINK_MODULE_FIELD_CASE, RM_LINK_MODULE_FIELD_CASE, GET_MODULES_FIELDS_CASE, SESSIONS_FIELDS_CASE,
]

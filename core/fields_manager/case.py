from .fields_lib import (
    handle_del_field, handle_set_field, handle_link_module_field,
    handle_list_modules_fields, handle_list_users_fields,
    handle_rm_link_module_field, handle_get_modules_fields,
    handle_get_sessions_fields
)

RELAY_FIELD = [
    {"case": "DEL_FIELD", "desc": "Delete Field", "func": handle_del_field, "req_struct": {"auth": {"field_id": "str", "user_id": "str"}}, "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": "list"}}},
    {"case": "LIST_USERS_FIELDS", "desc": "List Users Fields", "func": handle_list_users_fields, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": "list"}}},
    {"case": "LIST_MODULES_FIELDS", "desc": "List Modules Fields", "func": handle_list_modules_fields, "req_struct": {"auth": {"module_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": "list"}}},
    {"case": "SET_FIELD", "desc": "Set Field", "func": handle_set_field, "req_struct": {"data": {"field": "dict"}, "auth": {"user_id": "str", "original_id": "str"}}, "out_struct": {"type": "LIST_USERS_FIELDS", "data": {"fields": "list"}}},
    {"case": "LINK_MODULE_FIELD", "desc": "Link Module Field", "func": handle_link_module_field, "req_struct": {"auth": {"user_id": "str", "module_id": "str", "field_id": "str"}}, "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": "list"}}},
    {"case": "RM_LINK_MODULE_FIELD", "desc": "Remove Link Module Field", "func": handle_rm_link_module_field, "req_struct": {"auth": {"user_id": "str", "module_id": "str", "field_id": "str"}}, "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": "list"}}},
    {"case": "GET_MODULES_FIELDS", "desc": "Get Modules Fields", "func": handle_get_modules_fields, "req_struct": {"auth": {"user_id": "str", "module_id": "str"}}, "out_struct": {"type": "GET_MODULES_FIELDS", "data": {"fields": "list"}}},
    {"case": "SESSIONS_FIELDS", "desc": "Get Sessions Fields", "func": handle_get_sessions_fields, "req_struct": {"auth": {"user_id": "str", "session_id": "str"}}, "out_struct": {"type": "SESSIONS_FIELDS", "data": {"fields": "list"}}},
]

from .params_lib import (
    handle_get_users_params, handle_set_param, handle_del_param,
    handle_link_field_param, handle_rm_link_field_param, handle_get_fields_params
)

RELAY_PARAM = [
    {"case": "LIST_USERS_PARAMS", "desc": "Get Users Params", "func": handle_get_users_params, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "LIST_USERS_PARAMS", "data": {"params": "list"}}},
    {"case": "SET_PARAM", "desc": "Set Param", "func": handle_set_param, "req_struct": {"auth": {"user_id": "str"}, "data": {"param": "dict"}}, "out_struct": {"type": "LIST_USERS_PARAMS", "data": {"params": "list"}}},
    {"case": "DEL_PARAM", "desc": "Delete Param", "func": handle_del_param, "req_struct": {"auth": {"user_id": "str", "param_id": "str"}}, "out_struct": {"type": "LIST_USERS_PARAMS", "data": {"params": "list"}}},
    {"case": "LINK_FIELD_PARAM", "desc": "Link Field Param", "func": handle_link_field_param, "req_struct": {"auth": {"user_id": "str"}, "data": {"links": "list"}}, "out_struct": {"type": "GET_FIELDS_PARAMS", "data": {"params": "list"}}},
    {"case": "RM_LINK_FIELD_PARAM", "desc": "Rm Link Field Param", "func": handle_rm_link_field_param, "req_struct": {"auth": {"user_id": "str", "field_id": "str", "param_id": "str"}}, "out_struct": {"type": "GET_FIELDS_PARAMS", "data": {"params": "list"}}},
    {"case": "GET_FIELDS_PARAMS", "desc": "Get Fields Params", "func": handle_get_fields_params, "req_struct": {"auth": {"user_id": "str", "field_id": "str"}}, "out_struct": {"type": "GET_FIELDS_PARAMS", "data": {"params": "list"}}},
]

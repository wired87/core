from core.env_manager.env_lib import handle_link_env_module, handle_rm_link_env_module
from .modules_lib import (
    handle_del_module, handle_link_session_module, handle_rm_link_session_module,
    handle_set_module, handle_get_module, handle_get_sessions_modules,
    handle_list_users_modules
)

RELAY_MODULE = [
    {"case": "DEL_MODULE", "desc": "Delete Module", "func": handle_del_module, "req_struct": {"auth": {"module_id": "str", "user_id": "str"}}, "out_struct": {"type": "LIST_USERS_MODULES", "data": {"modules": "list"}}},
    {"case": "LINK_SESSION_MODULE", "desc": "Link Session Module", "func": handle_link_session_module, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "module_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_MODULES", "data": {"modules": "list"}}},
    {"case": "RM_LINK_SESSION_MODULE", "desc": "Remove Link Session Module", "func": handle_rm_link_session_module, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "module_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_MODULES", "data": {"modules": "list"}}},
    {"case": "LINK_ENV_MODULE", "desc": "Link Env Module", "func": handle_link_env_module, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "env_id": "str", "module_id": "str"}}, "out_struct": {"type": "LINK_ENV_MODULE", "data": {"sessions": "dict"}}},
    {"case": "RM_LINK_ENV_MODULE", "desc": "Remove Link Env Module", "func": handle_rm_link_env_module, "req_struct": {"auth": {"user_id": "str", "session_id": "str", "env_id": "str", "module_id": "str"}}, "out_struct": {"type": "LINK_ENV_MODULE", "data": {"sessions": "dict"}}},
    {"case": "SET_MODULE", "desc": "Set Module", "func": handle_set_module, "req_struct": {"data": {"id": "str", "files": "list", "methods": "list"}, "auth": {"user_id": "str"}}, "out_struct": {"type": "LIST_USERS_MODULES", "data": {"modules": "list"}}},
    {"case": "GET_MODULE", "desc": "Get Module", "func": handle_get_module, "req_struct": {"auth": {"module_id": "str"}}, "out_struct": {"type": "GET_MODULE", "data": "dict"}},
    {"case": "GET_SESSIONS_MODULES", "desc": "Get Session Modules", "func": handle_get_sessions_modules, "req_struct": {"auth": {"user_id": "str", "session_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_MODULES", "data": {"modules": "list"}, "auth": {"session_id": "str"}}},
    {"case": "LIST_USERS_MODULES", "desc": "List User Modules", "func": handle_list_users_modules, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "LIST_USERS_MODULES", "data": {"modules": "list"}}},
    {"case": "CONVERT_MODULE", "desc": "Convert Module", "func": None, "func_name": "_handle_convert_module", "req_struct": {"auth": {"module_id": "str"}, "data": {"files": "dict"}}, "out_struct": {"type": "CONVERT_MODULE", "data": "dict"}},
]

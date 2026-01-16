from .injection import (
    handle_set_inj, handle_del_inj, handle_get_inj_user, handle_get_inj_list,
    handle_link_inj_env, handle_rm_link_inj_env, handle_list_link_inj_env,
    handle_get_injection, handle_link_session_injection,
    handle_rm_link_session_injection, handle_get_sessions_injections
)

RELAY_INJECTION = [
    # INJECTION - Energy Designer
    {"case": "SET_INJ", "desc": "Set/upsert injection", "func": handle_set_inj, "req_struct": {"data": "dict", "auth": {"user_id": "str"}}, "out_struct": {"type": "GET_INJ_USER", "data": dict[str, list]}},
    {"case": "DEL_INJ", "desc": "Delete injection", "func": handle_del_inj, "req_struct": {"auth": {"injection_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_USER", "data": dict[str, list]}},
    {"case": "GET_INJ_USER", "desc": "Get user injections", "func": handle_get_inj_user, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "GET_INJ_USER", "data": dict[str, list]}},
    {"case": "GET_INJ_LIST", "desc": "Get injection list", "func": handle_get_inj_list, "req_struct": {"data": {"inj_ids": "list"}, "auth": {"user_id": "str"}}, "out_struct": {"type": "GET_INJ_LIST", "data": "list"}},
    
    # INJECTION - Env Linking
    {"case": "LINK_INJ_ENV", "desc": "Link injection to env", "func": handle_link_inj_env, "req_struct": {"auth": {"injection_id": "str", "env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_ENV", "data": "list"}},
    {"case": "RM_LINK_INJ_ENV", "desc": "Remove link injection to env", "func": handle_rm_link_inj_env, "req_struct": {"auth": {"injection_id": "str", "env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_ENV", "data": "list"}},
    {"case": "LIST_LINK_INJ_ENV", "desc": "List env linked injections", "func": handle_list_link_inj_env, "req_struct": {"auth": {"env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_ENV", "data": "list"}},
    {"case": "GET_INJ_ENV", "desc": "Get env injections", "func": handle_list_link_inj_env, "req_struct": {"auth": {"env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_INJ_ENV", "data": "list"}},
    {"case": "GET_INJECTION", "desc": "Get single injection", "func": handle_get_injection, "req_struct": {"auth": {"injection_id": "str"}}, "out_struct": {"type": "GET_INJECTION", "data": "dict"}},

    # INJECTIONS (Session)
    {"case": "LINK_SESSION_INJECTION", "desc": "Link Session Injection", "func": handle_link_session_injection, "req_struct": {"auth": {"session_id": "str", "injection_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_INJECTIONS", "data": {"injections": "list"}}},
    {"case": "RM_LINK_SESSION_INJECTION", "desc": "Remove Link Session Injection", "func": handle_rm_link_session_injection, "req_struct": {"auth": {"session_id": "str", "injection_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_INJECTIONS", "data": {"injections": "list"}}},
    {"case": "GET_SESSIONS_INJECTIONS", "desc": "Get Session Injections", "func": handle_get_sessions_injections, "req_struct": {"auth": {"session_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_SESSIONS_INJECTIONS", "data": {"injections": "list"}}},

    {"case": "REQUEST_INJ_SCREEN", "desc": "Requesting admin_data relavant for inj setup", "func": None, "func_name": "request_inj_process_start", "req_struct": {"data": {"env_id": "str"}}, "out_struct": {"type": "INJ_PATTERN_STRUCT", "admin_data": "dict", "env_id": "str"}},
    {"case": "SET_INJ_PATTERN", "desc": "Set ncfg injection pattern", "func": None, "func_name": "set_env_inj_pattern", "req_struct": {"data": "dict"}, "out_struct": None},
    {"case": "GET_INJ", "desc": "Retrieve inj cfg list", "func": None, "func_name": "set_env_inj_pattern", "req_struct": {"data": "dict"}, "out_struct": None},
]

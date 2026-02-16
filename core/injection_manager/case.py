from .injection import (
    handle_set_inj, handle_del_inj, handle_get_inj_user, handle_get_inj_list,
    handle_link_inj_env, handle_rm_link_inj_env, handle_list_link_inj_env,
    handle_get_injection, handle_link_session_injection,
    handle_rm_link_session_injection, handle_get_sessions_injections
)
from .types import (
    RelayCaseStruct,
)

# Case structs - req_struct.data uses exact datatypes (InjectionItemData)
SET_INJ_CASE: RelayCaseStruct = {
    "case": "SET_INJ", "desc": "Set/upsert injection", "func": handle_set_inj,
    "req_struct": {
        "data": {"id": str, "data": list, "ntype": str},  # InjectionItemData
        "auth": {"user_id": str, "original_id": str}
    },
    "out_struct": {"type": "GET_INJ_USER", "data": {"injections": list}},  # OutGetInjUser
}
DEL_INJ_CASE: RelayCaseStruct = {
    "case": "DEL_INJ", "desc": "Delete injection", "func": handle_del_inj,
    "req_struct": {"auth": {"injection_id": str, "user_id": str}},
    "out_struct": {"type": "GET_INJ_USER", "data": {"injections": list}},  # OutGetInjUser
}
GET_INJ_USER_CASE: RelayCaseStruct = {
    "case": "GET_INJ_USER", "desc": "Get user injections", "func": handle_get_inj_user,
    "req_struct": {"auth": {"user_id": str}},
    "out_struct": {"type": "GET_INJ_USER", "data": {"injections": list}},  # OutGetInjUser
}
GET_INJ_LIST_CASE: RelayCaseStruct = {
    "case": "GET_INJ_LIST", "desc": "Get injection list", "func": handle_get_inj_list,
    "req_struct": {"data": {"inj_ids": list}, "auth": {"user_id": str}},  # ReqDataGetInjList
    "out_struct": {"type": "GET_INJ_LIST", "data": list},  # OutGetInjList
}
LINK_INJ_ENV_CASE: RelayCaseStruct = {
    "case": "LINK_INJ_ENV", "desc": "Link injection to env", "func": handle_link_inj_env,
    "req_struct": {"auth": {"injection_id": str, "env_id": str, "user_id": str}},
    "out_struct": {"type": "GET_INJ_ENV", "data": list},  # OutGetInjEnv
}
RM_LINK_INJ_ENV_CASE: RelayCaseStruct = {
    "case": "RM_LINK_INJ_ENV", "desc": "Remove link injection to env", "func": handle_rm_link_inj_env,
    "req_struct": {"auth": {"injection_id": str, "env_id": str, "user_id": str}},
    "out_struct": {"type": "GET_INJ_ENV", "data": list},  # OutGetInjEnv
}
LIST_LINK_INJ_ENV_CASE: RelayCaseStruct = {
    "case": "LIST_LINK_INJ_ENV", "desc": "List env linked injections", "func": handle_list_link_inj_env,
    "req_struct": {"auth": {"env_id": str, "user_id": str}},
    "out_struct": {"type": "GET_INJ_ENV", "data": list},  # OutGetInjEnv
}
GET_INJECTION_CASE: RelayCaseStruct = {
    "case": "GET_INJECTION", "desc": "Get single injection", "func": handle_get_injection,
    "req_struct": {"data": {"id": str, "injection_id": str}, "auth": dict},  # ReqDataGetInjection
    "out_struct": {"type": "GET_INJECTION", "data": dict},  # OutGetInjection
}

# Subset for RELAY_CASES_CONFIG in predefined_case (core injection cases)
RELAY_INJECTION = [
    SET_INJ_CASE, DEL_INJ_CASE, GET_INJ_USER_CASE, GET_INJ_LIST_CASE,
    LINK_INJ_ENV_CASE, RM_LINK_INJ_ENV_CASE, LIST_LINK_INJ_ENV_CASE, GET_INJECTION_CASE,
]

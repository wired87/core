from .env_lib import (
    handle_get_env, handle_del_env, handle_set_env, handle_get_envs_user,
    handle_download_model, handle_retrieve_logs_env, handle_get_env_data
)

RELAY_ENV = [
    {"case": "GET_ENV", "desc": "", "func": handle_get_env, "req_struct": {"auth": {"env_id": "str"}}, "out_struct": {"type": "GET_ENV", "data": {"env": "dict"}}},
    {"case": "GET_USERS_ENVS", "desc": "", "func": handle_get_envs_user, "req_struct": {"auth": {"user_id": "str"}}, "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": "list"}}},
    {"case": "DEL_ENV", "desc": "", "func": handle_del_env, "req_struct": {"auth": {"env_id": "str", "user_id": "str"}}, "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": "list"}}},
    {"case": "SET_ENV", "desc": "", "func": handle_set_env, "req_struct": {"data": {"env_item": "dict"}, "auth": {"user_id": "str"}}, "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": "list"}}},
    {"case": "DOWNLOAD_MODEL", "desc": "Download Model", "func": handle_download_model, "req_struct": {"auth": {"env_id": "string", "user_id": "string"}}, "out_struct": {"type": "DOWNLOAD_MODEL", "data": "dict"}},
    {"case": "RETRIEVE_LOGS_ENV", "desc": "Retrieve Logs Env", "func": handle_retrieve_logs_env, "req_struct": {"auth": {"env_id": "string", "user_id": "string"}}, "out_struct": {"type": "RETRIEVE_LOGS_ENV", "data": "list"}},
    {"case": "GET_ENV_DATA", "desc": "Get Env Data", "func": handle_get_env_data, "req_struct": {"auth": {"env_id": "string", "user_id": "string"}}, "out_struct": {"type": "GET_ENV_DATA", "data": "list"}},
]

from .env_lib import (
    handle_get_env, handle_del_env, handle_set_env, handle_get_envs_user,
    handle_download_model, handle_retrieve_logs_env, handle_get_env_data
)
from .types import (
    EnvItemData, ReqDataSetEnv,
    OutGetEnv, OutGetUsersEnvs, OutDownloadModel, OutRetrieveLogsEnv, OutGetEnvData,
    RelayCaseStruct,
)

# req_struct.data uses exact datatypes from types.py (EnvItemData, ReqDataSetEnv)
SET_ENV_CASE: RelayCaseStruct = {
    "case": "SET_ENV", "desc": "", "func": handle_set_env,
    "req_struct": {
        "data": {"env": {"id": str, "sim_time": int, "cluster_dim": int, "dims": int, "data": str}},  # ReqDataSetEnv
        "auth": {"user_id": str, "original_id": str}
    },
    "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": list}},  # OutGetUsersEnvs
}

# Case structs with typed data sections
GET_ENV_CASE: RelayCaseStruct = {
    "case": "GET_ENV", "desc": "", "func": handle_get_env,
    "req_struct": {"auth": {"env_id": str}},
    "out_struct": {"type": "GET_ENV", "data": {"env": dict}},  # OutGetEnv
}
GET_USERS_ENVS_CASE: RelayCaseStruct = {
    "case": "GET_USERS_ENVS", "desc": "", "func": handle_get_envs_user,
    "req_struct": {"auth": {"user_id": str}},
    "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": list}},  # OutGetUsersEnvs
}
DEL_ENV_CASE: RelayCaseStruct = {
    "case": "DEL_ENV", "desc": "", "func": handle_del_env,
    "req_struct": {"auth": {"env_id": str, "user_id": str}},
    "out_struct": {"type": "GET_USERS_ENVS", "data": {"envs": list}},  # OutGetUsersEnvs
}
DOWNLOAD_MODEL_CASE: RelayCaseStruct = {
    "case": "DOWNLOAD_MODEL", "desc": "Download Model", "func": handle_download_model,
    "req_struct": {"auth": {"env_id": str, "user_id": str}},
    "out_struct": {"type": "DOWNLOAD_MODEL", "data": dict},  # OutDownloadModel
}
RETRIEVE_LOGS_ENV_CASE: RelayCaseStruct = {
    "case": "RETRIEVE_LOGS_ENV", "desc": "Retrieve Logs Env", "func": handle_retrieve_logs_env,
    "req_struct": {"auth": {"env_id": str, "user_id": str}},
    "out_struct": {"type": "RETRIEVE_LOGS_ENV", "data": list},  # OutRetrieveLogsEnv
}
GET_ENV_DATA_CASE: RelayCaseStruct = {
    "case": "GET_ENV_DATA", "desc": "Get Env Data", "func": handle_get_env_data,
    "req_struct": {"auth": {"env_id": str, "user_id": str}},
    "out_struct": {"type": "GET_ENV_DATA", "data": list},  # OutGetEnvData
}

RELAY_ENV = [
    GET_ENV_CASE, GET_USERS_ENVS_CASE, DEL_ENV_CASE, SET_ENV_CASE,
    DOWNLOAD_MODEL_CASE, RETRIEVE_LOGS_ENV_CASE, GET_ENV_DATA_CASE,
]

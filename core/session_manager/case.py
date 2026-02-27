from .session import (
    handle_link_env_session, handle_rm_link_env_session,
    handle_get_sessions_envs, handle_list_user_sessions
)
from .types import OutLinkEnvSession, OutGetSessionsEnvs, OutListUsersSessions, RelayCaseStruct

# Case structs with typed data sections
LINK_ENV_SESSION_CASE: RelayCaseStruct = {
    "case": "LINK_ENV_SESSION", "desc": "Link Env Session", "func": handle_link_env_session,
    "req_struct": {"auth": {"user_id": str, "session_id": str, "env_id": str}},
    "out_struct": {"type": "LINK_ENV_SESSION", "data": {"sessions": dict}},  # OutLinkEnvSession
}
RM_LINK_ENV_SESSION_CASE: RelayCaseStruct = {
    "case": "RM_LINK_ENV_SESSION", "desc": "Remove Link Env Session", "func": handle_rm_link_env_session,
    "req_struct": {"auth": {"user_id": str, "session_id": str, "env_id": str}},
    "out_struct": {"type": "RM_LINK_ENV_SESSION", "auth": {"session_id": str, "env_id": str}, "data": dict},  # Aligned with frontend
}
GET_SESSIONS_ENVS_CASE: RelayCaseStruct = {
    "case": "GET_SESSIONS_ENVS", "desc": "Get Session Envs", "func": handle_get_sessions_envs,
    "req_struct": {"auth": {"user_id": str, "session_id": str}},
    "out_struct": {"type": "GET_SESSIONS_ENVS", "data": {"envs": list}},  # OutGetSessionsEnvs
}
LIST_USERS_SESSIONS_CASE: RelayCaseStruct = {
    "case": "LIST_USERS_SESSIONS", "desc": "List user sessions", "func": handle_list_user_sessions,
    "req_struct": {"auth": {"user_id": str}},
    "out_struct": {"type": "LIST_USERS_SESSIONS", "data": {"sessions": list}},  # OutListUsersSessions
}

RELAY_SESSION = [
    LINK_ENV_SESSION_CASE, RM_LINK_ENV_SESSION_CASE, GET_SESSIONS_ENVS_CASE, LIST_USERS_SESSIONS_CASE,
]

from .sm_manager import handle_enable_sm

RELAY_SM = [
    {"case": "ENABLE_SM", "desc": "Enable Standard Model", "func": handle_enable_sm, "req_struct": {"auth": {"env_id": "str", "session_id": "str", "user_id": "str"}}, "out_struct": {"type": "ENABLE_SM", "data": {"sessions": "dict"}}},
]

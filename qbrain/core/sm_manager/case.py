from .sm_manager import handle_enable_sm
from .types import OutEnableSm, RelayCaseStruct

# Case struct with typed data section
ENABLE_SM_CASE: RelayCaseStruct = {
    "case": "ENABLE_SM", "desc": "Enable Standard Model", "func": handle_enable_sm,
    "req_struct": {"auth": {"env_id": str, "session_id": str, "user_id": str}},
    "out_struct": {"type": "ENABLE_SM", "data": {"sessions": dict}},  # OutEnableSm
}

RELAY_SM = [ENABLE_SM_CASE]

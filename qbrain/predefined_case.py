"""
RELAY_CASES_CONFIG: Manually composed from case structs exported by each manager's case module.
Each manager defines types in types.py and case structs in case.py with typed data sections.
"""
from qbrain.core.env_manager.case import RELAY_ENV
from qbrain.core.file_manager import RELAY_FILE
from qbrain.core.fields_manager.case import RELAY_FIELD
from qbrain.core.injection_manager.case import RELAY_INJECTION
from qbrain.core.method_manager.case import RELAY_METHOD
from qbrain.core.module_manager.ws_modules_manager.case import RELAY_MODULE
from qbrain.core.param_manager.case import RELAY_PARAM
from qbrain.core.session_manager.case import RELAY_SESSION

# Manually composed RELAY_CASES_CONFIG from case structs
# Order: ENV, FIELD, INJECTION, SESSION, MODULE, PARAMS, METHOD, FILE
RELAY_CASES_CONFIG = [
    *RELAY_ENV,
    *RELAY_FIELD,
    *RELAY_INJECTION,
    *RELAY_SESSION,
    *RELAY_MODULE,
    *RELAY_PARAM,
    *RELAY_METHOD,
    *RELAY_FILE,
]

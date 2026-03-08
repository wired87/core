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
from qbrain.control_engine import RELAY_CONTROL_ENGINE
from qbrain.core.collector_manager.case import RELAY_COLLECT_INFORMATION
from qbrain.core.sim_analyzer.case import RELAY_ANALYZE_SIM_RESULTS
from qbrain.core.researcher2.case import RELAY_START_RESEARCH

# Manually composed RELAY_CASES_CONFIG from case structs
# Order: COLLECT_INFORMATION, ANALYZE_SIM_RESULTS, START_RESEARCH, then ENV, FIELD, ...
RELAY_CASES_CONFIG = [
    *RELAY_COLLECT_INFORMATION,
    *RELAY_ANALYZE_SIM_RESULTS,
    *RELAY_START_RESEARCH,
    *RELAY_ENV,
    *RELAY_FIELD,
    *RELAY_INJECTION,
    *RELAY_SESSION,
    *RELAY_MODULE,
    *RELAY_PARAM,
    *RELAY_METHOD,
    *RELAY_FILE,
    *RELAY_CONTROL_ENGINE,
]

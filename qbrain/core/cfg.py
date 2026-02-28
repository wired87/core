from typing import Callable



def case_handling() -> dict[str, Callable]:
    from qbrain.core.module_manager.converter import Converter
    return {
        "convert_modules": Converter,
    }

RELAY_CONFIG = {
    "read_convert_modules": {
        "action": "Process of starting and converting all module files specified in arsenal dir."
    },
    "ask_question": {

    },
    "set_inj_schema": {

    },
    "start_sim": {
        "description": "Process of starting and converting all module files specified in arsenal dir.",
        "action": "create_guard"
    }
}


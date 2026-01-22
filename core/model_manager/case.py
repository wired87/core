from .model_lib import (
    handle_query_model
)

RELAY_MODEL = [
    {
        "case": "QUERY_MODEL",
        "desc": "Query the model structure using Gemini",
        "func": handle_query_model,
        "req_struct": {
            "auth": {
                "user_id": "str",
                "env_id": "str"
            },
            "data": {
                "question": "str"
            }
        },
        "out_struct": {
            "type": "QUERY_MODEL_RESPONSE",
            "data": {
                "answer": "str",
                "env_id": "str"
            }
        }
    }
]

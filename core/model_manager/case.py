from .model_lib import handle_query_model
from .types import ReqDataQueryModel, OutQueryModel, RelayCaseStruct

# Case struct - req_struct.data uses exact datatypes (ReqDataQueryModel)
QUERY_MODEL_CASE: RelayCaseStruct = {
    "case": "QUERY_MODEL",
    "desc": "Query the model structure using Gemini",
    "func": handle_query_model,
    "req_struct": {
        "auth": {"user_id": str, "env_id": str},
        "data": {"question": str}  # ReqDataQueryModel
    },
    "out_struct": {
        "type": "QUERY_MODEL_RESPONSE",
        "data": {"answer": str, "env_id": str}  # OutQueryModel
    }
}

RELAY_MODEL = [QUERY_MODEL_CASE]

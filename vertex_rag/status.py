from typing import Any, Dict


def build_success_response(case: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper to standardise Relay-style success responses for vertex_rag.
    """
    return {
        "type": case,
        "status": {
            "state": "success",
            "code": 200,
            "msg": "",
        },
        "data": data,
    }


def build_error_response(case: str, error: Exception) -> Dict[str, Any]:
    """
    Helper to standardise Relay-style error responses for vertex_rag.
    """
    return {
        "type": case,
        "status": {
            "state": "error",
            "code": 500,
            "msg": str(error),
        },
        "data": {},
    }


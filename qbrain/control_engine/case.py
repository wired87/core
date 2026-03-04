from __future__ import annotations

from typing import Any, Dict, List

from qbrain.core.env_manager.types import RelayCaseStruct

from .object_registry import list_spawnable_objects
from .spawn_manager import handle_spawn_object


def _handle_get_available_objects(*, data: Dict[str, Any], auth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Relay handler for GET_AVAILABLE_OBJECTS.

    Returns AVAILABLE_OBJECTS with a list of spawnable objects.
    """
    env_id = auth.get("env_id")
    user_id = auth.get("user_id")
    objects = list_spawnable_objects(env_id=str(env_id) if env_id is not None else None)
    return {
        "type": "AVAILABLE_OBJECTS",
        "status": {"state": "success", "code": 200, "msg": ""},
        "data": {
            "env_id": env_id,
            "user_id": user_id,
            "objects": objects,
        },
    }


GET_AVAILABLE_OBJECTS_CASE: RelayCaseStruct = {
    "case": "GET_AVAILABLE_OBJECTS",
    "desc": "List spawnable objects for control engine.",
    "func": _handle_get_available_objects,
    "req_struct": {
        "auth": {"user_id": str, "env_id": str},
        "data": {},
    },
    "out_struct": {
        "type": "AVAILABLE_OBJECTS",
        "data": {"objects": list},
    },
}


SPAWN_OBJECT_CASE: RelayCaseStruct = {
    "case": "SPAWN_OBJECT",
    "desc": "Spawn an object instance in a given environment.",
    "func": handle_spawn_object,
    "req_struct": {
        "auth": {"user_id": str, "env_id": str},
        "data": {
            "object_id": str,
            "position": {"x": float, "y": float, "z": float},
        },
    },
    "out_struct": {
        "type": "SPAWN_OBJECT_ACK",
        "data": {
            "env_id": str,
            "user_id": str,
            "instance_id": str,
            "object_id": str,
            "position": {"x": float, "y": float, "z": float},
        },
    },
}


RELAY_CONTROL_ENGINE: List[RelayCaseStruct] = [
    GET_AVAILABLE_OBJECTS_CASE,
    SPAWN_OBJECT_CASE,
]


from __future__ import annotations

import math
import uuid
from typing import Any, Dict, TypedDict, Optional

from .object_registry import get_object_meta


class SpawnPosition(TypedDict):
    x: float
    y: float
    z: float


WORLD_BOUNDS = {
    "x": (-50.0, 50.0),
    "y": (0.0, 50.0),
    "z": (-50.0, 50.0),
}


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _normalize_position(raw: Dict[str, Any]) -> SpawnPosition:
    x = _coerce_float(raw.get("x"), 0.0)
    y = _coerce_float(raw.get("y"), 0.0)
    z = _coerce_float(raw.get("z"), 0.0)
    xmin, xmax = WORLD_BOUNDS["x"]
    ymin, ymax = WORLD_BOUNDS["y"]
    zmin, zmax = WORLD_BOUNDS["z"]
    return {
        "x": max(xmin, min(xmax, x)),
        "y": max(ymin, min(ymax, y)),
        "z": max(zmin, min(zmax, z)),
    }


def handle_spawn_object(*, data: Dict[str, Any], auth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Relay handler for SPAWN_OBJECT.

    Expects:
        auth: { "env_id": str, "user_id": str }
        data: { "object_id": str, "position": {x,y,z} }

    Returns SPAWN_OBJECT_ACK with normalized position and instance_id.
    """
    env_id = str(auth.get("env_id") or "").strip()
    user_id = str(auth.get("user_id") or "").strip()
    object_id = str(data.get("object_id") or "").strip()

    if not env_id or not user_id:
        return {
            "type": "SPAWN_OBJECT_ACK",
            "status": {
                "state": "error",
                "code": 400,
                "msg": "Missing env_id or user_id in auth for SPAWN_OBJECT.",
            },
            "data": {},
        }

    if not object_id:
        return {
            "type": "SPAWN_OBJECT_ACK",
            "status": {
                "state": "error",
                "code": 400,
                "msg": "Missing object_id for SPAWN_OBJECT.",
            },
            "data": {"env_id": env_id},
        }

    meta = get_object_meta(object_id)
    if meta is None:
        return {
            "type": "SPAWN_OBJECT_ACK",
            "status": {
                "state": "error",
                "code": 404,
                "msg": f"Unknown object_id={object_id!r} for SPAWN_OBJECT.",
            },
            "data": {"env_id": env_id, "object_id": object_id},
        }

    raw_position = data.get("position") or {}
    position: SpawnPosition = _normalize_position(raw_position)

    # For now we keep registration lightweight: instance id + echo of visual/default_state
    instance_id = f"{object_id}::{uuid.uuid4().hex[:8]}"

    # TODO: integrate with EnvManager / Guard grid representation once schema is stable.
    return {
        "type": "SPAWN_OBJECT_ACK",
        "status": {
            "state": "success",
            "code": 200,
            "msg": "",
        },
        "data": {
            "env_id": env_id,
            "user_id": user_id,
            "instance_id": instance_id,
            "object_id": object_id,
            "position": position,
            "visual": meta.get("visual", {}),
            "default_state": meta.get("default_state", {}),
        },
    }


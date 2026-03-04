from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, Literal, Any


ObjectType = Literal["particle", "field", "constraint"]


class ObjectVisualMeta(TypedDict, total=False):
    shape: str
    color: str
    size: float


class ObjectMeta(TypedDict, total=False):
    id: str
    label: str
    type: ObjectType
    default_state: Dict[str, Any]
    visual: ObjectVisualMeta


@dataclass
class ObjectRegistry:
    """
    In‑memory registry of spawnable objects for the control engine.

    This is intentionally simple and non‑persistent for now; objects can later be
    hydrated from QBrain tables / EnvManager once the contract is stable.
    """

    _objects: Dict[str, ObjectMeta]

    @classmethod
    def default(cls) -> "ObjectRegistry":
        # Minimal demo catalogue; can be replaced by DB‑backed loader later.
        base: List[ObjectMeta] = [
            {
                "id": "particle_basic",
                "label": "Basic Particle",
                "type": "particle",
                "default_state": {"mass": 1.0, "charge": 0.0},
                "visual": {"shape": "sphere", "color": "#22d3ee", "size": 1.0},
            },
            {
                "id": "field_scalar",
                "label": "Scalar Field",
                "type": "field",
                "default_state": {"intensity": 1.0},
                "visual": {"shape": "box", "color": "#a78bfa", "size": 1.5},
            },
            {
                "id": "constraint_fixed",
                "label": "Fixed Constraint",
                "type": "constraint",
                "default_state": {"locked": True},
                "visual": {"shape": "rect", "color": "#f97316", "size": 1.2},
            },
        ]
        return cls(_objects={obj["id"]: obj for obj in base})

    def list_objects(self, env_id: Optional[str] = None) -> List[ObjectMeta]:
        """
        Return all spawnable objects. The env_id parameter is accepted to allow
        future environment‑specific filtering but is currently ignored.
        """
        # Hook for per‑environment filtering in the future.
        return list(self._objects.values())

    def get(self, object_id: str) -> Optional[ObjectMeta]:
        return self._objects.get(object_id)


_DEFAULT_REGISTRY: Optional[ObjectRegistry] = None


def get_registry() -> ObjectRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ObjectRegistry.default()
    return _DEFAULT_REGISTRY


def list_spawnable_objects(env_id: Optional[str] = None) -> List[ObjectMeta]:
    """
    Convenience wrapper used by WebSocket handlers.
    """
    return get_registry().list_objects(env_id=env_id)


def get_object_meta(object_id: str) -> Optional[ObjectMeta]:
    """
    Convenience wrapper used by the SpawnManager.
    """
    return get_registry().get(object_id)


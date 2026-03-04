"""
Control engine backend utilities for QDash.

This package exposes Relay/WebSocket case definitions for:

- GET_AVAILABLE_OBJECTS
- SPAWN_OBJECT

These cases are consumed by the Thalamus orchestrator and wired into the
existing RELAY_CASES_CONFIG registry.
"""

from .case import RELAY_CONTROL_ENGINE

__all__ = ["RELAY_CONTROL_ENGINE"]


"""
CreateEnvFromComponents: Build env from field_ids, method_ids, module_id.

Guided flow for "start sim" -> collect fields, methods -> create envs.
Every method follows: print("method_name...") at start, print("method_name... done") at end.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from qbrain.core.qbrain_manager import get_qbrain_table_manager


# ---- Public API ----

def create_env_from_components(
    user_id: str,
    module_id: str,
    field_ids: List[str],
    method_ids: List[str],
    session_id: Optional[str] = None,
    env_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create or update an env with modules, fields, and methods.

    Builds env_data.modules structure expected by ComponentGraphCreator.
    Links env to session when session_id provided.

    Args:
        user_id: User owning the env.
        module_id: Module to attach to env.
        field_ids: Field IDs to include (linked to module).
        method_ids: Method IDs to include (linked to module).
        session_id: Optional session to link env to.
        env_id: Optional existing env ID to update; else creates new.

    Returns:
        Dict with env_id, env_data, status, error (if any).
    """
    print("create_env_from_components...")
    qb = get_qbrain_table_manager()
    env_table = "envs"
    env_id = env_id or f"env_{random.randint(1000000000, 9999999999)}"

    # Build modules structure: {module_id: {fields: {field_id: {}}}}
    fields_cfg = {fid.upper(): {} for fid in (field_ids or [])}
    modules_data = {module_id.upper(): {"fields": fields_cfg, "methods": method_ids or []}}

    env_data = {"modules": modules_data}

    try:
        row = {
            "id": env_id,
            "user_id": user_id,
            "data": json.dumps(env_data),
            "status": "active",
        }
        qb.set_item(env_table, row, keys={"id": env_id, "user_id": user_id})
    except Exception as e:
        print(f"create_env_from_components: set_item error: {e}")
        print("create_env_from_components... done")
        return {"env_id": env_id, "status": "error", "error": str(e), "env_data": env_data}

    if session_id:
        try:
            _link_env_session(qb, session_id, env_id, user_id)
            _link_env_module(qb, session_id, env_id, module_id, user_id)
            for fid in (field_ids or []):
                _link_module_field(qb, session_id, env_id, module_id, fid, user_id)
        except Exception as e:
            print(f"create_env_from_components: link error: {e}")

    print("create_env_from_components... done")
    return {"env_id": env_id, "env_data": env_data, "status": "created"}


# ---- Internal helpers ----

def _link_env_session(qb: Any, session_id: str, env_id: str, user_id: str) -> None:
    """Link env to session in session_to_envs table."""
    print("_link_env_session...")
    row_id = str(random.randint(1000000000, 9999999999))
    row = {"id": row_id, "session_id": session_id, "env_id": env_id, "user_id": user_id}
    qb.set_item("session_to_envs", row)
    print("_link_env_session... done")


def _link_env_module(qb: Any, session_id: str, env_id: str, module_id: str, user_id: str) -> None:
    """Link module to env in envs_to_modules table."""
    print("_link_env_module...")
    row_id = str(random.randint(1000000000, 9999999999))
    row = {
        "id": row_id,
        "session_id": session_id,
        "env_id": env_id,
        "module_id": module_id,
        "user_id": user_id,
    }
    try:
        qb.set_item("envs_to_modules", row)
    except Exception as e:
        print(f"_link_env_module: {e}")
    print("_link_env_module... done")


def _link_module_field(
    qb: Any,
    session_id: str,
    env_id: str,
    module_id: str,
    field_id: str,
    user_id: str,
) -> None:
    """Link field to module in modules_to_fields table."""
    print("_link_module_field...")
    row_id = str(random.randint(1000000000, 9999999999))
    row = {
        "id": row_id,
        "session_id": session_id,
        "env_id": env_id,
        "module_id": module_id,
        "field_id": field_id.upper(),
        "user_id": user_id,
    }
    try:
        qb.set_item("modules_to_fields", row)
    except Exception as e:
        print(f"_link_module_field: {e}")
    print("_link_module_field... done")


def validate_env_components(env_data: Optional[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """
    Validate env_data has required modules/fields for simulation.

    Returns:
        (is_valid, missing_components) where missing_components are human-readable strings.
    """
    print("validate_env_components...")
    missing: List[str] = []
    if not env_data:
        missing.append("env_data")
        print("validate_env_components... done")
        return False, missing

    modules = env_data.get("modules")
    if modules is None:
        missing.append("modules")
    elif isinstance(modules, (list, dict)) and len(modules) == 0:
        missing.append("modules (at least one)")

    if isinstance(modules, dict):
        for mid, mcfg in modules.items():
            fields = mcfg.get("fields") if isinstance(mcfg, dict) else {}
            if not fields:
                missing.append(f"fields for module {mid}")

    is_valid = len(missing) == 0
    print("validate_env_components... done")
    return is_valid, missing

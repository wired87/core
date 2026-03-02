"""
Build LIVE_DATA payload from current time_db state: dict[keys, shaped_param] (JSON-serializable).
Uses param_ctlr + keys list and reshape logic from visualizer.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .visualizer import _reshape_flat_db

DEFAULT_SCHEMA = {
    "param_ctrl": "DB_PARAM_CONTROLLER",
    "amount_per_field": "AMOUNT_PARAMS_PER_FIELD",
    "modules": "MODULES",
    "fields": "FIELDS",
    "db_keys": "DB_KEYS",
    "field_keys": "FIELD_KEYS",
}


def _flat_keys_from_cfg(cfg: dict[str, Any], schema: dict[str, str]) -> list[str]:
    """Produce a flat list of key names in same order as reshaped params (module, field, param)."""
    param_ctrl = cfg.get(schema["param_ctrl"]) or []
    amount_per_field = cfg.get(schema["amount_per_field"]) or []
    modules = cfg.get(schema["modules"]) or [0]
    fields = cfg.get(schema["fields"]) or [1]
    n_modules = max(1, max(modules) + 1) if modules else 1
    n_fields = max(1, max(fields)) if fields else 1
    db_keys = cfg.get(schema["db_keys"])
    field_keys = cfg.get(schema["field_keys"])
    keys = []
    idx = 0
    for _mi in range(n_modules):
        for _fi in range(n_fields):
            n_params = amount_per_field[_mi * n_fields + _fi] if (_mi * n_fields + _fi) < len(amount_per_field) else 1
            for pi in range(n_params):
                if db_keys and idx < len(db_keys):
                    keys.append(str(db_keys[idx]))
                elif field_keys and idx < len(field_keys):
                    keys.append(str(field_keys[idx]))
                else:
                    keys.append(f"p_{idx}")
                idx += 1
    return keys


def build_live_data_payload(
    cfg: dict[str, Any],
    time_db_flat: np.ndarray,
    keys: list[str] | None = None,
    schema: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Convert time_db[0] (flat array) to dict[key_name, list[float]] using param_ctlr and keys.
    Returns JSON-serializable dict suitable for LIVE_DATA message.
    """
    schema = {**DEFAULT_SCHEMA, **(schema or {})}
    param_ctrl = cfg.get(schema["param_ctrl"]) or []
    amount_per_field = cfg.get(schema["amount_per_field"]) or []
    modules = cfg.get(schema["modules"]) or [0]
    fields = cfg.get(schema["fields"]) or [1]
    n_modules = max(1, max(modules) + 1) if modules else 1
    n_fields = max(1, max(fields)) if fields else 1

    flat = np.asarray(time_db_flat)
    if np.iscomplexobj(flat):
        flat = np.abs(flat.astype(np.complex64)).astype(np.float32)
    else:
        flat = flat.astype(np.float32)
    flat = np.ravel(flat)

    reshaped = _reshape_flat_db(flat, list(param_ctrl), list(amount_per_field), n_modules, n_fields)
    key_list = keys if keys is not None else _flat_keys_from_cfg(cfg, schema)

    out = {}
    idx = 0
    for mod in reshaped:
        for field in mod:
            for param_arr in field:
                k = key_list[idx] if idx < len(key_list) else f"p_{idx}"
                out[k] = param_arr.ravel().tolist()
                idx += 1
    return out

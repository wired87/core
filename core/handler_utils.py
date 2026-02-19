"""
Shared utilities for manager handler methods.
Handlers receive data and auth from payload (no flattening).
"""


def get_val(data: dict, auth: dict, key: str, default=None):
    """Get value from auth or data. Auth takes precedence."""
    if isinstance(auth, dict) and key in auth and auth[key] is not None and auth[key] != "":
        return auth[key]
    return data.get(key, default) if isinstance(data, dict) else default


def flatten_payload(payload: dict) -> dict:
    """
    Flatten nested payload {auth: {...}, data: {...}} to top-level dict.
    All keys from auth and data become top-level kwargs for handler methods.
    """
    if not isinstance(payload, dict):
        return {}
    auth = payload.get("auth") or {}
    data = payload.get("data") or {}
    if not isinstance(auth, dict):
        auth = {}
    if not isinstance(data, dict):
        data = {}
    return {**auth, **data}


def param_missing_error(key: str) -> dict:
    """Return standard error object for missing required param."""
    return {"error": "param missing", "key": key}


def require_param(val, key: str) -> dict | None:
    """
    Check if required param is present. Returns error dict if missing, else None.
    Use: if err := require_param(user_id, "user_id"): return err
    """
    if val is None or val == "":
        return param_missing_error(key)
    return None


def require_param_truthy(val, key: str) -> dict | None:
    """For required dict/list - also rejects empty {} or []."""
    if val is None or val == "" or (isinstance(val, (list, dict)) and not val):
        return param_missing_error(key)
    return None

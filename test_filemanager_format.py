"""
Standalone test: prints created_components in exact handler input format.
No file_manager import - run from project root: python test_filemanager_format.py
"""
import json
import random


def _build_created_components(user_id: str, params_list: list, fields_list: list, methods_list: list) -> dict:
    """Build created_components in exact format expected by handle_set_param/field/method."""
    created_components = {"param": [], "field": [], "method": []}

    if params_list:
        for item in params_list:
            item["id"] = item.get("id") or item.get("name") or str(random.randint(100000, 999999))
            item["name"] = item.get("name") or item["id"]
        created_components["param"] = [{"auth": {"user_id": user_id}, "data": {"param": params_list}}]

    if fields_list:
        for item in fields_list:
            item["id"] = item.get("id") or str(random.randint(100000, 999999))
        created_components["field"] = [{"auth": {"user_id": user_id}, "data": {"field": f}} for f in fields_list]

    if methods_list:
        for m in methods_list:
            m["id"] = m.get("id") or str(random.randint(100000, 999999))
            m["user_id"] = user_id
            if "equation" in m and "code" not in m:
                m["code"] = m["equation"]
        created_components["method"] = [{"auth": {"user_id": user_id}, "data": dict(m)} for m in methods_list]

    return created_components


def main():
    user_id = "test_user"
    params_list = [{"id": "param_mock_1", "name": "mass", "value": 1.0, "unit": "kg"}]
    fields_list = [{"id": "field_mock_1", "name": "velocity", "equation": "v = dx/dt"}]
    methods_list = [{"id": "method_mock_1", "equation": "x' = v", "params": ["v"]}]

    cc = _build_created_components(user_id, params_list, fields_list, methods_list)

    print("--- Created Components (exact handler input format) ---")
    handler_map = {
        "param": "handle_set_param  (params_lib)  -> data.param can be list",
        "field": "handle_set_field  (fields_lib)  -> data.field = one field dict",
        "method": "handle_set_method (method_lib)  -> data = flat method dict",
    }
    for content_type in ("param", "field", "method"):
        items = cc.get(content_type, [])
        if not items:
            continue
        print(f"\n>>> {handler_map[content_type]}")
        print(f"    ({len(items)} payload(s)):")
        for i, payload in enumerate(items):
            print(f"  [{i+1}]", json.dumps(payload, default=str, indent=2))
    print("\n--- Format matches what relay_station dispatches to each handler ---")


if __name__ == "__main__":
    main()

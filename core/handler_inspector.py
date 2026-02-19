"""
Inspector to programmatically extract handler functions from manager modules.
Uses AST to pick up handle_* functions (excludes methods with self param),
captures docstring as description, full source code, and module imports.
"""
import ast
import inspect
import os
from typing import List, Dict, Any, Optional




def _get_module_source(module_name: str) -> Optional[str]:
    """Load module source from file path."""
    try:
        import importlib.util
        parts = module_name.split(".")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rel_path = os.path.join(*parts) + ".py"
        file_path = os.path.join(project_root, rel_path)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"[HandlerInspector] _get_module_source {module_name}: {e}")
    return None


def _extract_imports_from_source(source: str) -> str:
    """Extract all import statements from source (lines until first non-import)."""
    imports = []
    in_import_block = True
    for line in source.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if imports:
                imports.append(line)
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
            in_import_block = True
        elif in_import_block and imports:
            break
    return "\n".join(imports) if imports else ""


def _has_self_param(node: ast.FunctionDef) -> bool:
    """True if function has 'self' as first param (i.e. is a method)."""
    if node.args.args:
        return node.args.args[0].arg == "self"
    return False


def extract_handlers_from_module(module_name: str, source: str) -> List[Dict[str, Any]]:
    """
    Parse module source and extract handle_* functions (exclude methods with self).
    Returns list of dicts: nid (func name), description (docstring), code (full def), imports.
    """
    entries = []
    try:
        tree = ast.parse(source)
        module_imports = _extract_imports_from_source(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("handle_"):
                    continue
                if _has_self_param(node):
                    continue
                docstring = ast.get_docstring(node) or ""
                lines = source.splitlines()
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
                code = "\n".join(lines[start:end]) if lines else ""
                entries.append({
                    "nid": node.name,
                    "description": docstring.strip(),
                    "code": code,
                    "imports": module_imports,
                    "module_id": module_name,
                })
    except Exception as e:
        print(f"[HandlerInspector] extract_handlers_from_module {module_name}: {e}")
    return entries


def collect_all_handlers() -> List[Dict[str, Any]]:
    """Collect handler entries from all HANDLER_MODULES."""
    all_entries = []
    for mod in HANDLER_MODULES:
        source = _get_module_source(mod)
        if source:
            entries = extract_handlers_from_module(mod, source)
            all_entries.extend(entries)
    return all_entries


def register_handlers_to_gutils(g_utils) -> None:
    """
    Use StructInspector-style logic: add each handler as node to g_utils.G.
    nid=func_name, description=docstring, code=full_def, imports=module_imports.
    """
    entries = collect_all_handlers()
    for entry in entries:
        attrs = {
            "nid": entry["nid"],
            "type": "HANDLER",
            "description": entry["description"],
            "code": entry["code"],
            "imports": entry["imports"],
            "module_id": entry["module_id"],
        }
        try:
            g_utils.add_node(attrs=attrs)
        except Exception as exc:
            print(f"[HandlerInspector] add_node {entry['nid']}: {exc}")

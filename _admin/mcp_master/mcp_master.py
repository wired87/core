"""
MCP Master: scans qbrain for case.py RELAY_* definitions, exposes HTTP routes.

Relocated from qbrain.bm.mcp_converter to _admin for centralized admin tooling.
"""
import ast
import importlib
import importlib.util
import os
import pprint
from pathlib import Path
from typing import Any, Dict


def _resolve_qbrain_root() -> str:
    """Resolve qbrain package root (parent of bm)."""
    # When imported from qbrain.bm.settings, __file__ is _admin/mcp_master/mcp_master.py
    this = Path(__file__).resolve()
    # _admin/mcp_master -> _admin -> project root
    project_root = this.parent.parent.parent
    qbrain = project_root / "qbrain"
    return str(qbrain) if qbrain.is_dir() else str(project_root)


class MCPMaster:
    def __init__(self):
        self.rbp = None
        self.relay_cfg: Dict[str, Any] = {}

        self.server_struct_out = r"C:\Users\bestb\PycharmProjects\BestBrain\_admin\mcp_master\mcp_server"

    def main(self) -> None:
        print("MCP conversion...")
        self.relay_cfg = self.scan_cases_and_update_relay()
        self.expose_api()
        print("MCP conversion... done")

    def set_route_blueprint(self):
        # define route
        from rest_framework.views import APIView
        from requests import Response

        class RouteBlueprint(APIView):
            """POST mcp/{sub_route}/{case} — invoke relay case handler."""

            # case_struct passed via as_view(case_struct=item); View.__init__ sets it

            def post(self, request, *args, **kwargs):
                try:
                    response = self.case_struct["func"](*request.data)
                    return Response(response)
                except Exception as e:
                    return Response({"error": str(e)})

        return RouteBlueprint

    def expose_api(self) -> None:
        from django.conf import settings
        from django.urls import path
        print("expose_api...")
        server_struct = []

        urls = importlib.import_module(settings.ROOT_URLCONF)
        RouteView = self.set_route_blueprint()

        for sub_route, items in self.relay_cfg.items():
            for item in items:
                route = f"mcp/{sub_route}/{item['case']}"
                urls.urlpatterns.append(
                    path(route, RouteView.as_view(case_struct=item))
                )

                server_struct.append(
                    {
                        "endpoint": route,
                        "req_struct": item["req_struct"],
                        "description": item["desc"],
                    }
                )


        print("expose_api... done")

    def scan_cases_and_update_relay(self, root_dir: str | None = None) -> Dict[str, Any]:
        if root_dir is None:
            root_dir = _resolve_qbrain_root()

        root_dir = os.path.normpath(root_dir)

        result: Dict[str, Any] = {
            "scanned": 0,
            "relay_cases_found": 0,
            "imported": 0,
            "errors": [],
        }
        relay_cfg: Dict[str, Any] = {}
        skip_dirs = {"__pycache__", ".git", "venv", ".venv", "node_modules"}

        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]

            for name in filenames:
                if name != "case.py":
                    continue

                filepath = os.path.join(dirpath, name)
                result["scanned"] += 1

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        code = f.read()
                except Exception as e:
                    result["errors"].append({"path": filepath, "error": str(e)})
                    continue

                try:
                    tree = ast.parse(code)
                except Exception as e:
                    result["errors"].append({"path": filepath, "error": f"AST parse error: {e}"})
                    continue

                relay_vars = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and "RELAY" in target.id:
                                relay_vars.append(target.id)

                if not relay_vars:
                    continue

                result["relay_cases_found"] += len(relay_vars)

                try:
                    spec = importlib.util.spec_from_file_location(
                        "relay_case_module", filepath
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    result["errors"].append({"path": filepath, "error": f"Import error: {e}"})
                    continue

                for var in relay_vars:
                    try:
                        value = getattr(module, var)
                        relay_cfg[var] = value
                        result["imported"] += 1
                    except Exception as e:
                        result["errors"].append(
                            {"path": filepath, "error": f"Variable load error {var}: {e}"}
                        )

        print("case structs found")
        pprint.pp(result)
        return relay_cfg

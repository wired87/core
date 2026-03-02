from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from qbrain.graph import GUtils


_COLLECTOR_DEBUG = "[CaseCollectorManager]"


@dataclass
class DiscoveredCase:
    """Metadata wrapper for a single relay case struct discovered in a case.py module."""

    case_name: str
    desc: str
    req_struct: Dict[str, Any]
    out_struct: Dict[str, Any]
    func_name: Optional[str]
    module_name: str
    file_path: str
    relay_group: str  # e.g. RELAY_ENV, RELAY_FIELD, ...
    index: int


class CaseCollectorManager:
    """
    CaseCollectorManager scans the codebase for manager `case.py` files that define
    relay case structs (RELAY_* lists of dicts with a "case" key) and mirrors
    each discovered case struct into nodes on a provided `GUtils` graph helper.

    Typical usage:

        collector = CaseCollectorManager(gutils=brain)
        count = collector.collect_cases_into_graph()

    This is intentionally read‑only w.r.t. application behaviour: it does not
    modify RELAY_CASES_CONFIG or handler wiring; it only adds descriptive nodes
    so other components (e.g. Brain / tooling) can introspect available cases.
    """

    def __init__(
        self,
        gutils: GUtils,
        project_root: Optional[str] = None,
    ) -> None:
        """
        Args:
            gutils: Target `GUtils` instance; discovered cases are added as nodes
                on its underlying NetworkX graph.
            project_root: Root directory to start scanning from. Defaults to
                the qbrain package root.
        """
        self.gutils = gutils
        self.G: Optional[nx.Graph] = getattr(gutils, "G", None)
        if project_root is None:
            # qbrain/core/collector_manager -> qbrain
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )
        self.project_root = project_root

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def collect_cases_into_graph(self) -> int:
        """
        Discover all relay cases from manager case.py modules and add them as
        nodes to self.G.

        Returns:
            Number of nodes added or updated.
        """
        if self.G is None:
            print(f"{_COLLECTOR_DEBUG} collect_cases_into_graph: no graph on GUtils, aborting")
            return 0

        discovered = list(self._discover_relay_cases())
        count = 0
        for case in discovered:
            node_id = self._build_node_id(case)
            attrs = {
                "id": node_id,
                "type": "CASE",
                "case": case.case_name,
                "desc": case.desc,
                "req_struct": case.req_struct,
                "out_struct": case.out_struct,
                "func_name": case.func_name,
                "source_module": case.module_name,
                "source_file": case.file_path,
                "relay_group": case.relay_group,
                "relay_index": case.index,
            }
            self._upsert_node(node_id, attrs)
            count += 1

        print(f"{_COLLECTOR_DEBUG} collect_cases_into_graph: added/updated {count} CASE nodes")
        return count

    # --------------------------------------------------------------------- #
    # Discovery
    # --------------------------------------------------------------------- #
    def _discover_relay_cases(self) -> Iterable[DiscoveredCase]:
        """
        Yield DiscoveredCase instances by scanning all case.py modules for
        RELAY_* lists that contain dicts with a "case" key.
        """
        print(f"{_COLLECTOR_DEBUG} discover_relay_cases: scanning from {self.project_root}")
        for file_path in self._iter_case_files():
            module_name = self._module_name_from_path(file_path)
            try:
                for group_name, case_defs in self._load_relay_groups(file_path, module_name):
                    for idx, case_def in enumerate(case_defs):
                        case = self._build_discovered_case(
                            case_def=case_def,
                            module_name=module_name,
                            file_path=file_path,
                            relay_group=group_name,
                            index=idx,
                        )
                        if case is not None:
                            yield case
            except Exception as e:
                print(f"{_COLLECTOR_DEBUG} discover_relay_cases: skip {file_path}: {e}")

    def _iter_case_files(self) -> Iterable[str]:
        """Yield absolute paths to all case.py files under project_root."""
        exclude_dirs = {
            ".venv",
            "venv",
            ".git",
            "__pycache__",
            ".idea",
            "node_modules",
            "build",
            "dist",
        }
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for fname in files:
                if fname != "case.py":
                    continue
                yield os.path.join(root, fname)

    def _module_name_from_path(self, file_path: str) -> str:
        """
        Derive a pseudo‑module name from a file path relative to project_root.
        This is primarily for diagnostics / graph metadata and does not have
        to match the importable module path exactly.
        """
        rel = os.path.relpath(file_path, self.project_root)
        # Normalize os separators and strip .py
        return os.path.splitext(rel.replace(os.sep, "."))[0]

    def _load_relay_groups(
        self,
        file_path: str,
        module_name: str,
    ) -> Iterable[Tuple[str, List[Dict[str, Any]]]]:
        """
        Import a case.py module and yield (group_name, list_of_case_dicts) for
        each attribute that looks like a RELAY_* = [ {…}, … ] definition.
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            return []
        module = importlib.util.module_from_spec(spec)
        # Register temporarily so relative imports inside the module work.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[assignment]

        results: List[Tuple[str, List[Dict[str, Any]]]] = []
        for attr_name, value in vars(module).items():
            if not attr_name.startswith("RELAY_"):
                continue
            # We only care about manager case structs that are lists of dicts with a "case" key.
            if not isinstance(value, list) or not value:
                continue
            if not isinstance(value[0], dict) or "case" not in value[0]:
                continue
            # At this point we assume value is List[RelayCaseStruct].
            results.append((attr_name, value))  # type: ignore[arg-type]
        return results

    def _build_discovered_case(
        self,
        case_def: Dict[str, Any],
        module_name: str,
        file_path: str,
        relay_group: str,
        index: int,
    ) -> Optional[DiscoveredCase]:
        """Convert a raw case_def dict into a DiscoveredCase, or None if invalid."""
        try:
            case_name = str(case_def.get("case") or "").strip()
            if not case_name:
                return None
            desc = str(case_def.get("desc") or "")
            req_struct = case_def.get("req_struct") or {}
            out_struct = case_def.get("out_struct") or {}

            func = case_def.get("func")
            func_name: Optional[str] = None
            if callable(func):
                func_name = getattr(func, "__name__", None) or str(func)
            else:
                # Some case structs use a "func_name" indirection (e.g. CONVERT_MODULE_CASE).
                fn = case_def.get("func_name")
                if isinstance(fn, str) and fn:
                    func_name = fn

            return DiscoveredCase(
                case_name=case_name,
                desc=desc,
                req_struct=req_struct,
                out_struct=out_struct,
                func_name=func_name,
                module_name=module_name,
                file_path=file_path,
                relay_group=relay_group,
                index=index,
            )
        except Exception as e:
            print(f"{_COLLECTOR_DEBUG} build_discovered_case: skip invalid case in {file_path}: {e}")
            return None

    # --------------------------------------------------------------------- #
    # Graph helpers
    # --------------------------------------------------------------------- #
    def _build_node_id(self, case: DiscoveredCase) -> str:
        """
        Build a stable node identifier for a discovered case. Names are scoped
        by module to avoid collisions between managers.
        """
        return f"CASE::{case.module_name}::{case.case_name}"

    def _upsert_node(self, node_id: str, attrs: Dict[str, Any]) -> None:
        """Add or update a CASE node on the target graph."""
        try:
            # Keep a copy without the "id" attribute for networkx; store "id"
            # as a separate attribute for consistency with other graph code.
            node_attrs = dict(attrs)
            node_attrs.pop("id", None)
            if self.G.has_node(node_id):
                self.G.nodes[node_id].update(node_attrs)
            else:
                self.G.add_node(node_id, **node_attrs)
        except Exception as e:
            print(f"{_COLLECTOR_DEBUG} upsert_node: error for {node_id}: {e}")


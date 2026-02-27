"""
Test for the case pickup process: scan core/ dir, convert all .py modules to graph,
pick up all methods, and print the graph structure.

Run from project root: py core/test_case_pickup.py
"""
import os
import sys

# Ensure project root is on path
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# Minimal setup - avoid full Relay init for faster test
from graph.local_graph_utils import GUtils
from code_manipulation.graph_creator import StructInspector


def scan_dir_to_code_graph(root_dir: str, cg: GUtils, inspector: StructInspector) -> dict:
    """Scan directory, convert .py files to graph. Returns result stats."""
    root_dir = os.path.normpath(root_dir)
    result = {"scanned": 0, "converted": 0, "errors": []}
    skip_dirs = {"__pycache__", ".git", "venv", ".venv", "node_modules"}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            filepath = os.path.join(dirpath, name)
            result["scanned"] += 1
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    code_content = f.read()
            except Exception as e:
                result["errors"].append({"path": filepath, "error": str(e)})
                continue

            rel = os.path.relpath(filepath, root_dir)
            module_name = rel[:-3].replace(os.sep, ".")

            try:
                inspector.convert_module_to_graph(code_content, module_name)
                result["converted"] += 1
            except Exception as e:
                result["errors"].append({"path": filepath, "error": str(e)})

    return result


def print_graph_structure(G):
    """Print graph structure: nodes by type, edges summary, sample node attrs."""
    nodes = list(G.nodes())
    node_data = {nid: dict(G.nodes[nid]) for nid in nodes}

    by_type = {}
    for nid, attrs in node_data.items():
        t = attrs.get("type", "UNKNOWN")
        by_type.setdefault(t, []).append(nid)

    print("\n" + "=" * 60)
    print("GRAPH STRUCTURE")
    print("=" * 60)
    print(f"\nTotal nodes: {len(nodes)}")
    print(f"Total edges: {len(G.edges())}")

    print("\n--- Nodes by type ---")
    for t in sorted(by_type.keys()):
        print(f"  {t}: {len(by_type[t])}")

    print("\n--- METHOD nodes (first 15) ---")
    methods = by_type.get("METHOD", [])
    for nid in sorted(methods)[:15]:
        attrs = node_data[nid]
        module = attrs.get("module_id", "?")
        desc = (attrs.get("docstring", "") or "")[:50]
        print(f"  {nid}  [module: {module}]  {desc}...")

    if len(methods) > 15:
        print(f"  ... and {len(methods) - 15} more")

    print("\n--- HANDLER nodes (from handler_inspector) ---")
    handlers = by_type.get("HANDLER", [])
    for nid in sorted(handlers)[:15]:
        attrs = node_data[nid]
        module = attrs.get("module_id", "?")
        desc = (attrs.get("description", "") or "")[:50]
        print(f"  {nid}  [module: {module}]  {desc}...")

    if len(handlers) > 15:
        print(f"  ... and {len(handlers) - 15} more")

    print("\n--- Edge types (src_layer -> trgt_layer) ---")
    edge_types = {}
    for src, trg, attrs in G.edges(data=True):
        sl = attrs.get("src_layer", "?")
        tl = attrs.get("trgt_layer", "?")
        key = f"{sl} -> {tl}"
        edge_types[key] = edge_types.get(key, 0) + 1
    for k, v in sorted(edge_types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)


def run_test():
    """Run case pickup test on core/ directory."""
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_dir = os.path.join(proj_root, "core")

    print("Case pickup test: scanning core/")
    print(f"Core dir: {core_dir}")

    cg = GUtils(nx_only=True, enable_data_store=False)
    inspector = StructInspector(cg.G)

    # Scan core dir first (METHOD nodes from StructInspector)
    result = scan_dir_to_code_graph(core_dir, cg, inspector)

    # Register handlers (HANDLER nodes - overwrites handle_* METHOD nodes with callable)
    from core.handler_inspector import register_handlers_to_gutils, collect_all_handlers
    handlers = collect_all_handlers()
    print(f"\nRegistering {len(handlers)} handlers...")
    register_handlers_to_gutils(cg)

    print(f"\nScan result: scanned={result['scanned']}, converted={result['converted']}, errors={len(result['errors'])}")
    if result["errors"]:
        for e in result["errors"][:5]:
            print(f"  Error: {e['path']}: {e['error']}")

    print_graph_structure(cg.G)
    


if __name__ == "__main__":
    run_test()

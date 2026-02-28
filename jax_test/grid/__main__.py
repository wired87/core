"""
JAX grid package entry point: python -m jax_test.grid [--cfg PATH] [--cfg-string JSON]

Loads cfg from file (--cfg) or from JSON string (--cfg-string), then runs the workflow
via jax_test.grid.Guard.
"""

import argparse
import json
import os
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid workflow: run with cfg from file or string")
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to cfg JSON file (e.g. test_out.json). Ignored if --cfg-string is set.",
    )
    parser.add_argument(
        "--cfg-string",
        type=str,
        default=None,
        metavar="JSON",
        help="Cfg as JSON string. If provided, overrides --cfg.",
    )
    return parser.parse_args()


def _load_cfg(args: argparse.Namespace) -> dict:
    if args.cfg_string is not None:
        try:
            return json.loads(args.cfg_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --cfg-string JSON: {e}") from e

    cfg_path = args.cfg
    if cfg_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cfg_path = os.path.join(project_root, "test_out.json")

    if not os.path.isabs(cfg_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cfg_path = os.path.join(project_root, cfg_path)

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Cfg file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = _parse_args()
    try:
        cfg = _load_cfg(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    from jax_test.grid.guard import Guard

    guard = Guard(cfg=cfg)
    guard.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

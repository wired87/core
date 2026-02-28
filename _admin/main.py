#!/usr/bin/env python
"""
Admin workflow entry point: run with `python -m _admin.main`.

- Discovers all project dirs (Dockerfile + package.json / manage.py / requirements.txt).
- Classifies each as backend_drf | backend_fastapi | backend_py | frontend_react | mobile_react_native.
- Builds all Dockerfile dirs (skips if image exists unless --force-rebuild).
- Builds frontend/mobile with npm run build when no Dockerfile.
- Deploys to local Kubernetes if LOCAL=true in project root .env, else to cloud (GKE from .env).
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env from project root
try:
    from dotenv import load_dotenv
    _env_path = _project_root / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from _admin.admin import Admin


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Admin: discover projects, classify (backend/frontend/mobile), build all Dockerfile dirs, deploy to K8s (local if LOCAL=true else cloud from .env)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Deploy to local Kubernetes cluster (current kubectl context); overrides .env LOCAL",
    )
    parser.add_argument("--no-build", action="store_true", help="Skip build step")
    parser.add_argument("--no-deploy", action="store_true", help="Skip deploy step")
    parser.add_argument("--no-push", action="store_true", help="Do not push to Artifact Registry (when not LOCAL)")
    parser.add_argument(
        "--no-npm-build",
        action="store_true",
        help="Do not run npm run build for frontend/mobile projects without Dockerfile",
    )
    parser.add_argument("--tag", default="latest", help="Docker image tag (default: latest)")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only print discovered and classified project dirs, then exit",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild Docker images even when they already exist locally",
    )
    # Run locally (no Docker build/run)
    parser.add_argument(
        "--run-local",
        action="store_true",
        help="Run discovered apps locally (no Docker build/run). Mutually exclusive with build/deploy.",
    )
    parser.add_argument(
        "--run-local-project",
        type=str,
        default=None,
        metavar="PATH",
        help="Run only this project (path relative to project root or absolute). With --run-local only.",
    )
    parser.add_argument(
        "--run-local-port",
        type=str,
        default=None,
        metavar="PORT_OR_MAP",
        help="Port or mapping for run-local (e.g. 8000 or backend_drf:8000,frontend:3000). Defaults: backend 8000, frontend 3000.",
    )
    parser.add_argument(
        "--run-local-scan-only",
        action="store_true",
        help="Print discovered projects with inferred type, start command, and cwd; no execution.",
    )
    args = parser.parse_args()

    project_root = _project_root
    if args.run_local or args.run_local_scan_only:
        from _admin.run_local import run_local_scan_only, run_local_execute
        if args.run_local_scan_only:
            run_local_scan_only(project_root)
            sys.exit(0)
        run_local_execute(
            project_root,
            project_path=args.run_local_project,
            port_spec=args.run_local_port,
        )
        sys.exit(0)

    admin = Admin(tag=args.tag, local=args.local if args.local else None)

    if args.scan_only:
        for path, image_name, project_type, has_docker in admin.scan_projects():
            try:
                rel = path.relative_to(admin.project_root)
            except ValueError:
                rel = path
            print(f"  {rel} -> image={image_name} type={project_type} dockerfile={has_docker}")
        sys.exit(0)

    out = admin.run(
        build=not args.no_build,
        push=None if not args.no_push else False,
        deploy=not args.no_deploy,
        namespace=args.namespace,
        build_frontend_npm=not args.no_npm_build,
        force_rebuild=args.force_rebuild,
    )
    print("[Admin] Built (images):", len(out["built"]))
    print("[Admin] Pushed:", len(out["pushed_uris"]))
    print("[Admin] Deploy results:", out["deploy_results"])


if __name__ == "__main__":
    main()

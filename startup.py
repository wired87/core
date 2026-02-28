#!/usr/bin/env python
"""
Unified engine startup: backend prep, qdash frontend build, nginx, and Daphne.

Usage:
  python startup.py              # Full startup
  python startup.py --skip-frontend   # Skip qdash build
  python startup.py --skip-nginx      # Skip nginx (e.g. Cloud Run)
  python startup.py --backend-only    # --skip-frontend --skip-nginx
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "qbrain.bm.settings")

# Load .env if present
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


def _log(msg: str) -> None:
    print(f"[startup] {msg}", flush=True)


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    _log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT)
    if check and result.returncode != 0:
        _log(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def run_migrations() -> None:
    _run([sys.executable, "manage.py", "migrate", "--noinput"])


def run_collectstatic() -> None:
    _run([sys.executable, "manage.py", "collectstatic", "--noinput", "--clear"])


def run_qdash_build(skip: bool) -> None:
    qdash_dir = PROJECT_ROOT / "qdash"
    build_dir = qdash_dir / "build"
    static_root = PROJECT_ROOT / "static_root"

    if skip:
        _log("Skipping qdash build (--skip-frontend)")
        if build_dir.is_dir():
            static_root.mkdir(parents=True, exist_ok=True)
            for item in build_dir.iterdir():
                dst = static_root / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
            _log(f"Copied pre-built qdash/build to {static_root}")
        return

    if not qdash_dir.is_dir():
        _log("qdash/ not found, skipping frontend build")
        return
    pkg = qdash_dir / "package.json"
    if not pkg.exists():
        _log("qdash/package.json not found, skipping frontend build")
        return

    lock = qdash_dir / "package-lock.json"
    if lock.exists():
        _run(["npm", "ci"], cwd=qdash_dir)
    else:
        _run(["npm", "install"], cwd=qdash_dir)
    _run(["npm", "run", "build"], cwd=qdash_dir)

    if not build_dir.is_dir():
        _log("qdash/build not found after build")
        return

    static_root.mkdir(parents=True, exist_ok=True)
    for item in build_dir.iterdir():
        dst = static_root / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    _log(f"Copied qdash/build to {static_root}")


def run_nginx(skip: bool) -> None:
    if skip:
        _log("Skipping nginx (--skip-nginx)")
        return
    _run([sys.executable, "-m", "qbrain.nginx.render_nginx_conf"])

    nginx_conf_dir = PROJECT_ROOT / "qbrain" / "nginx"
    confs = list(nginx_conf_dir.glob("*.conf"))
    if not confs:
        _log("No nginx/*.conf found after render")
        return

    dest = "/etc/nginx/sites-enabled/default"
    try:
        shutil.copy(confs[0], dest)
        _log(f"Copied {confs[0].name} to {dest}")
    except (PermissionError, OSError) as e:
        _log(f"Could not copy nginx config to {dest}: {e}")
        _log("Nginx may use a different config path")
        return

    try:
        subprocess.Popen(
            ["nginx", "-g", "daemon on;"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _log("Nginx started")
    except FileNotFoundError:
        _log("nginx not found in PATH")
    except Exception as e:
        _log(f"Nginx start error: {e}")


def run_daphne() -> None:
    port = os.environ.get("PORT", "8080")
    _log(f"Starting Daphne on 0.0.0.0:{port}")
    cmd = [
        sys.executable, "-m", "daphne",
        "--application-close-timeout", "30",
        "-b", "0.0.0.0",
        "-p", str(port),
        "qbrain.bm.asgi:application",
    ]
    os.execvp(sys.executable, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="BestBrain engine startup")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip qdash build")
    parser.add_argument("--skip-nginx", action="store_true", help="Skip nginx render and start")
    parser.add_argument("--backend-only", action="store_true", help="Equivalent to --skip-frontend --skip-nginx")
    args = parser.parse_args()

    skip_frontend = args.skip_frontend or args.backend_only
    skip_nginx = args.skip_nginx or args.backend_only

    _log("Pre-flight: PYTHONPATH ok")
    run_migrations()
    run_collectstatic()
    run_qdash_build(skip=skip_frontend)
    run_nginx(skip=skip_nginx)
    run_daphne()


if __name__ == "__main__":
    main()

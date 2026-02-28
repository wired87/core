"""
Run discovered projects locally without Docker.
Infers start commands from project type, startup.py, manage.py, main.py,
Dockerfile CMD/ENTRYPOINT, and package.json scripts.
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from _admin.project_discovery import discover_projects
from _admin.project_classifier import (
    BACKEND_DRF,
    BACKEND_FASTAPI,
    BACKEND_PY,
    FRONTEND_REACT,
    MOBILE_REACT_NATIVE,
    UNKNOWN,
)


def get_project_root() -> Path:
    """Project root (same as docker_scanner)."""
    return Path(__file__).resolve().parent.parent


def parse_dockerfile_cmd(dir_path: Path) -> list[str] | None:
    """
    Parse the last CMD or ENTRYPOINT from a Dockerfile in dir_path.
    Returns list of args (e.g. ["python", "startup.py", "--skip-frontend"]) or None.
    Handles CMD ["exec", "arg"] and CMD exec arg forms.
    """
    dockerfile = dir_path / "Dockerfile"
    if not dockerfile.is_file():
        return None
    try:
        content = dockerfile.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    # Find last CMD or ENTRYPOINT (CMD overrides ENTRYPOINT for run; we want what actually runs)
    last_cmd: list[str] | None = None
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # CMD ["a", "b"] or CMD ["a"]
        m = re.match(r'^\s*CMD\s+\[(.*)\]', line)
        if m:
            # Parse JSON-like list (simple: split by ", " and strip quotes)
            rest = m.group(1).strip()
            if not rest:
                last_cmd = []
            else:
                parts = re.findall(r'"([^"]*)"', rest)
                if not parts:
                    parts = re.findall(r"'([^']*)'", rest)
                last_cmd = parts if parts else [rest]
            continue
        m = re.match(r'^\s*ENTRYPOINT\s+\[(.*)\]', line)
        if m:
            rest = m.group(1).strip()
            parts = re.findall(r'"([^"]*)"', rest)
            if not parts:
                parts = re.findall(r"'([^']*)'", rest)
            last_cmd = parts if parts else [rest]
    return last_cmd


def _is_root_project(path: Path, project_root: Path) -> bool:
    try:
        return path.resolve() == project_root.resolve()
    except Exception:
        return False


def _is_qbrain_core(path: Path, project_root: Path) -> bool:
    """True if path is qbrain/core (VM bootstrap Dockerfile; not run as standalone)."""
    try:
        return path.resolve() == (project_root / "qbrain" / "core").resolve()
    except Exception:
        return False


def _is_qbrain_dir(path: Path, project_root: Path) -> bool:
    """True if path is the qbrain package dir (has bm/settings)."""
    try:
        return path.resolve() == (project_root / "qbrain").resolve() and (path / "bm" / "settings.py").exists()
    except Exception:
        return False


def infer_start_command(
    path: Path,
    project_type: str,
    project_root: Path,
    has_dockerfile: bool,
) -> tuple[list[str], Path, dict[str, str] | None]:
    """
    Infer (command_argv, cwd, env_override) for running this project locally.
    env_override can be None (use current env) or a dict to merge into os.environ.
    """
    path = path.resolve()
    project_root = project_root.resolve()
    env: dict[str, str] = {}
    docker_cmd = parse_dockerfile_cmd(path) if has_dockerfile else None

    # qbrain/core: VM bootstrap only; do not run standalone
    if _is_qbrain_core(path, project_root):
        return ([], path, None)  # Caller will treat empty cmd as "skip"

    # Root project with startup.py (monolith)
    if _is_root_project(path, project_root) and (project_root / "startup.py").is_file():
        cmd = [sys.executable, "startup.py", "--backend-only"]
        return (cmd, project_root, {"PYTHONPATH": str(project_root), "DJANGO_SETTINGS_MODULE": "qbrain.bm.settings"})

    # backend_drf (Django): run from project root with daphne
    if project_type == BACKEND_DRF:
        if _is_root_project(path, project_root):
            if (project_root / "startup.py").is_file():
                cmd = [sys.executable, "startup.py", "--backend-only"]
            else:
                cmd = [
                    sys.executable, "-m", "daphne",
                    "-b", "0.0.0.0", "-p", "8000",
                    "qbrain.bm.asgi:application",
                ]
            env = {"PYTHONPATH": str(project_root), "DJANGO_SETTINGS_MODULE": "qbrain.bm.settings"}
            return (cmd, project_root, env)
        if _is_qbrain_dir(path, project_root):
            # qbrain package: run from root, same as root backend
            cmd = [
                sys.executable, "-m", "daphne",
                "-b", "0.0.0.0", "-p", "8000",
                "qbrain.bm.asgi:application",
            ]
            env = {"PYTHONPATH": str(project_root), "DJANGO_SETTINGS_MODULE": "qbrain.bm.settings"}
            return (cmd, project_root, env)
        # Other DRF (e.g. nested): try manage.py runserver or daphne from that dir
        manage = path / "manage.py"
        if manage.is_file():
            cmd = [sys.executable, "manage.py", "runserver", "0.0.0.0:8000"]
            env = {"PYTHONPATH": str(path)}
            return (cmd, path, env)
        return ([], path, None)

    # backend_fastapi
    if project_type == BACKEND_FASTAPI:
        for name in ("main.py", "app.py"):
            f = path / name
            if f.is_file():
                mod = f.stem
                cmd = [sys.executable, "-m", "uvicorn", f"{mod}:app", "--host", "0.0.0.0", "--port", "8000"]
                return (cmd, path, {"PYTHONPATH": str(path)})
        return ([], path, None)

    # backend_py: Dockerfile CMD or main.py / startup.py
    if project_type == BACKEND_PY:
        if docker_cmd:
            # Adapt paths: Docker often uses /app; we use path
            cmd = []
            for part in docker_cmd:
                if part == "python":
                    cmd.append(sys.executable)
                elif part.endswith(".py") and not Path(part).is_absolute():
                    cmd.append(str(path / part))
                else:
                    cmd.append(part)
            if cmd:
                env = {"PYTHONPATH": str(path)}
                return (cmd, path, env)
        for name in ("startup.py", "main.py", "app.py"):
            f = path / name
            if f.is_file():
                cmd = [sys.executable, str(f)]
                return (cmd, path, {"PYTHONPATH": str(path)})
        # jax_test default
        if (path / "test_gnn_run.py").is_file():
            return ([sys.executable, "test_gnn_run.py"], path, {"PYTHONPATH": str(path)})
        return ([], path, None)

    # frontend_react / mobile_react_native
    if project_type in (FRONTEND_REACT, MOBILE_REACT_NATIVE):
        pkg = path / "package.json"
        if pkg.is_file():
            cmd = ["npm", "run", "start"]
            return (cmd, path, None)
        return ([], path, None)

    return ([], path, None)


def is_standalone_runnable(path: Path, project_type: str, project_root: Path) -> bool:
    """True if this project should be run as its own process (not skipped)."""
    if _is_qbrain_core(path, project_root):
        return False
    return True


def run_local_scan_only(project_root: Path | None = None) -> None:
    """Print discovered projects with inferred type, start command, and cwd."""
    root = project_root or get_project_root()
    projects = discover_projects(root)
    print("[Admin run-local] Discovered projects (inferred command, cwd):")
    for path, image_name, project_type, has_docker in projects:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        cmd, cwd, env = infer_start_command(path, project_type, root, has_docker)
        if not is_standalone_runnable(path, project_type, root):
            print(f"  {rel} -> type={project_type} dockerfile={has_docker} -> skip (VM/bootstrap)")
            continue
        if not cmd:
            print(f"  {rel} -> type={project_type} dockerfile={has_docker} -> (no inferred command)")
            continue
        try:
            cwd_rel = cwd.relative_to(root)
        except ValueError:
            cwd_rel = cwd
        cmd_str = " ".join(cmd)
        print(f"  {rel} -> type={project_type} dockerfile={has_docker}")
        print(f"      cwd={cwd_rel}  cmd={cmd_str}")


def _parse_port_spec(port_spec: str | None) -> dict[str, int]:
    """Parse --run-local-port: single number or backend_drf:8000,frontend:3000."""
    default_backend = 8000
    default_frontend = 3000
    if not port_spec or not port_spec.strip():
        return {"backend": default_backend, "frontend": default_frontend}
    spec = port_spec.strip()
    if spec.isdigit():
        return {"backend": int(spec), "frontend": default_frontend}
    out: dict[str, int] = {"backend": default_backend, "frontend": default_frontend}
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            k, v = k.strip(), v.strip()
            if v.isdigit():
                out[k] = int(v)
    return out


def run_one(
    path: Path,
    project_type: str,
    project_root: Path,
    has_dockerfile: bool,
    port: int | None = None,
) -> subprocess.Popen[Any] | None:
    """
    Run a single project in a subprocess. Returns the Popen instance or None if skipped/failed.
    """
    if not is_standalone_runnable(path, project_type, project_root):
        print(f"[Admin run-local] Skip {path.name} (not standalone runnable)")
        return None
    cmd, cwd, env_override = infer_start_command(path, project_type, project_root, has_dockerfile)
    if not cmd:
        print(f"[Admin run-local] No inferred command for {path}")
        return None
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    if port is not None:
        env["PORT"] = str(port)
        # Replace -p / --port in cmd if present (daphne, uvicorn, etc.)
        for i, arg in enumerate(cmd):
            if arg == "-p" and i + 1 < len(cmd):
                cmd = cmd[: i + 1] + [str(port)] + cmd[i + 2:]
                break
            if arg == "--port" and i + 1 < len(cmd):
                cmd = cmd[: i + 1] + [str(port)] + cmd[i + 2:]
                break
    try:
        rel = path.relative_to(project_root)
    except ValueError:
        rel = path
    print(f"[Admin run-local] Starting {rel}: cwd={cwd} cmd={' '.join(cmd)}")
    try:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=False,
        )
        return p
    except Exception as e:
        print(f"[Admin run-local] Failed to start {rel}: {e}")
        return None


def _should_skip_because_root_backend(
    path: Path,
    project_type: str,
    project_root: Path,
    projects: list[tuple[Path, str, str, bool]],
) -> bool:
    """Skip qbrain when root is in the list and will run the same Django app."""
    if project_type != BACKEND_DRF:
        return False
    if not _is_qbrain_dir(path, project_root):
        return False
    root_has_startup = (project_root / "startup.py").is_file()
    root_in_list = any(_is_root_project(p, project_root) for p, _, _, _ in projects)
    return root_in_list and root_has_startup


def run_all(
    project_root: Path,
    project_path_filter: str | None = None,
    port_spec: str | None = None,
) -> list[subprocess.Popen[Any]]:
    """
    Run all runnable projects (or the one matching project_path_filter).
    Returns list of Popen instances. Caller may wait or run in foreground.
    Deduplicates root vs qbrain: only one Django backend run when both are discovered.
    """
    root = project_root.resolve()
    projects = discover_projects(root)
    port_map = _parse_port_spec(port_spec)
    backend_port = port_map.get("backend", 8000)
    frontend_port = port_map.get("frontend", 3000)

    if project_path_filter:
        filter_path = (root / project_path_filter).resolve() if not Path(project_path_filter).is_absolute() else Path(project_path_filter).resolve()
        projects = [(p, name, pt, h) for p, name, pt, h in projects if p.resolve() == filter_path]
        if not projects:
            print(f"[Admin run-local] No project found at {project_path_filter}")
            return []

    processes: list[subprocess.Popen[Any]] = []
    backend_count = 0
    frontend_count = 0

    for path, _name, project_type, has_docker in projects:
        if not is_standalone_runnable(path, project_type, root):
            continue
        if _should_skip_because_root_backend(path, project_type, root, projects):
            continue
        cmd, _cwd, _ = infer_start_command(path, project_type, root, has_docker)
        if not cmd:
            continue
        port: int | None = None
        if project_type in (BACKEND_DRF, BACKEND_FASTAPI, BACKEND_PY):
            port = backend_port + backend_count
            backend_count += 1
        elif project_type in (FRONTEND_REACT, MOBILE_REACT_NATIVE):
            port = frontend_port + frontend_count
            frontend_count += 1
        p = run_one(path, project_type, root, has_docker, port=port)
        if p is not None:
            processes.append(p)

    return processes


def run_local_execute(
    project_root: Path,
    project_path: str | None = None,
    port_spec: str | None = None,
) -> None:
    """
    Execute run-local: start processes and wait for them (Ctrl+C to stop all).
    """
    procs = run_all(project_root, project_path_filter=project_path, port_spec=port_spec)
    if not procs:
        print("[Admin run-local] No projects started.")
        return
    print(f"[Admin run-local] Started {len(procs)} process(es). Ctrl+C to stop.")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        print("[Admin run-local] Stopped.")

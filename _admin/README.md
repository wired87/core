# _admin

Admin CLI for BestBrain: discover projects, classify by type, build Docker images, deploy to K8s (local or GKE), and **run all apps locally** without Docker.

## Entrypoint

```bash
python -m _admin.main [options]
```

## Run locally (no Docker)

Run discovered apps natively by inferring start commands from project structure.

| Option | Description |
|--------|-------------|
| `--run-local` | Run all runnable projects (or the one given by `--run-local-project`). No build/deploy. |
| `--run-local-project PATH` | Run only this project (path relative to repo root, e.g. `qdash`, `jax_test`). |
| `--run-local-port PORT_OR_MAP` | Port or mapping, e.g. `8000` or `backend_drf:8000,frontend:3000`. Defaults: backend 8000, frontend 3000. |
| `--run-local-scan-only` | Print discovered projects with inferred type, command, and cwd; no execution. |

### Inferred commands and run context

| Project path | Type | Has Dockerfile | Inferred local run |
|--------------|------|----------------|--------------------|
| (root) | monolith | Yes | `python startup.py --backend-only` (cwd=root) |
| qbrain | backend_drf | Yes | Root cwd: `python -m daphne ... qbrain.bm.asgi:application` (or root `startup.py --backend-only`) |
| qdash | frontend_react | Yes | cwd=qdash: `npm run start` |
| jax_test | backend_py | Yes | cwd=jax_test: `python test_gnn_run.py` (from Dockerfile CMD) |
| qbrain/core | backend_* | Yes | **Skipped**: VM bootstrap (`startup.sh`), not a standalone app; use root backend. |

- **Root vs qbrain**: Only one Django process is started when both root and qbrain are discovered (root with `startup.py` takes precedence; qbrain is skipped to avoid duplicate backend).
- **Dockerfile CMD/ENTRYPOINT** is parsed and used as a fallback for backend_py (e.g. jax_test).
- Backend ports are assigned in order (8000, 8001, …); frontend ports (3000, 3001, …). Set `PORT` in the environment for frontends (e.g. Create React App respects it).

## Build and deploy (Docker + K8s)

| Option | Description |
|--------|-------------|
| `--scan-only` | Print discovered and classified project dirs (image name, type, has_dockerfile); exit. |
| `--local` | Deploy to local Kubernetes (current kubectl context). |
| `--no-build` | Skip build step. |
| `--no-deploy` | Skip deploy step. |
| `--no-push` | Do not push to Artifact Registry (GKE). |
| `--no-npm-build` | Do not run npm run build for frontend/mobile projects without Dockerfile. |
| `--tag TAG` | Docker image tag (default: latest). |
| `--namespace NS` | Kubernetes namespace (default: default). |
| `--force-rebuild` | Rebuild Docker images even when they already exist locally. |

Discovery uses `_admin.project_discovery` and `_admin.bob_builder.docker_scanner`; classification uses `_admin.project_classifier` (backend_drf, backend_fastapi, backend_py, frontend_react, mobile_react_native).

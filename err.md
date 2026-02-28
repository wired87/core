# _admin.main Workflow — Issues Log

**Goal:** Build all Docker engines and ensure stable execution.

---

## How to run

From project root (with Docker Desktop running if you want builds to succeed):

```bash
# List discovered projects (no Docker/k8s needed)
py -m _admin.main --scan-only

# Build all Dockerfile-based images (no push, no deploy)
py -m _admin.main --no-deploy --no-push

# Full pipeline: build, push to registry (if not LOCAL), deploy to K8s
py -m _admin.main

# Rebuild images even if they already exist
py -m _admin.main --force-rebuild --no-deploy
```

---

## Session 1

### Scan-only / discovery

- **Discovery (without full Admin):** OK. Root: `C:\Users\bestb\PycharmProjects\BestBrain`. Discovered 6 projects:
  - `BestBrain` (backend_drf, has Dockerfile)
  - `MiracleAI` (mobile_react_native, no Dockerfile)
  - `jax_test` (backend_py, has Dockerfile)
  - `qbrain` (backend_drf, has Dockerfile)
  - `core` (unknown, has Dockerfile) — **note:** `qbrain/core` has a Dockerfile; may be double-counted or path confusion
  - `qdash` (frontend_react, has Dockerfile)

- **Full `py -m _admin.main --scan-only`:** Was hanging; fixed by lazy-init of DockerAdmin/GkeDeployer (see Resolved).

---

## Resolved

1. **Scan-only hang:** Resolved by lazy-initializing `DockerAdmin` and `GkeDeployer` (and deferring their imports) so `--scan-only` does not load docker_admin (which pulls in qbrain) or create deployer. Scan-only now completes and prints 6 projects including 5 with Dockerfiles: root (BestBrain), jax_test, qbrain, qbrain\\core, qdash.
2. **qbrain build context:** Admin now passes `context_dir=project_root` when building the qbrain image so `qbrain/Dockerfile` gets the correct context (r.txt, manage.py, qbrain/).

---

### Build run (--no-deploy --no-push)

- **Docker daemon not running:** All 5 Docker builds failed with: `error during connect: Head "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.` **Mitigation:** Start Docker Desktop (or ensure Docker engine is running) before running the build. Not a code defect.
- **qbrain image build context:** `qbrain/Dockerfile` expects build context = **project root** (it copies `r.txt`, `manage.py`, `qbrain/`). **Fixed:** DockerAdmin accepts optional `context_dir`; Admin passes `context_dir=project_root` when building the `qbrain` image so the build uses project root as context.
- **MiracleAI:** `npm not found; skip npm build` — expected if npm not on PATH; no Dockerfile so no image produced.
- **Built (images): 0** because all Docker builds failed (daemon off). Workflow completed without crash.

---

## Open

1. **Environment:** Start Docker Desktop (or Docker engine) to actually build images.
2. Optionally exclude `qbrain/core` from Dockerfile scan if only the root `qbrain/` Dockerfile is desired.

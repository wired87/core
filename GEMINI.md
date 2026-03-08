# BestBrain – Integrations & Context for Gemini CLI

Case-driven orchestration engine for simulation workflows: WebSocket + HTTP, manager buses, QBRAIN-backed persistence. **qdash** (React frontend), **qbrain** (Django backend), **_admin** (tooling).

---

## Gemini CLI

### Settings

- **Config**: `.gemini/settings.json` (or root `settings.json`)
- **Regenerate**: `py -m _admin.main --write-gemini-settings`
- **Context files**: `GEMINI.md`, `CONTEXT.md`
- **Include dirs**: root, `qdash`, `qbrain`, `_admin`

### MCP Servers (Gemini CLI)

| Server | Env | Description |
|--------|-----|-------------|
| **github** | `GITHUB_PERSONAL_ACCESS_TOKEN` | List PRs, create issues, manage repos. Docker-based. |

Add token to `.env` or export before running Gemini CLI.

---

## QBrain Relay Extension (Gemini CLI)

Extension at `qbrain/mcpmaster/` exposes relay cases as tools.

### Env

- `GEMINI_API_KEY` – for extraction/enrichment
- `RELAY_WS_URL` – optional WebSocket URL

### Relay Cases (see `qbrain/mcpmaster/GEMINI.md`)

- **ENV**: GET_ENV, GET_USERS_ENVS, SET_ENV, DEL_ENV, DOWNLOAD_MODEL, RETRIEVE_LOGS_ENV, GET_ENV_DATA
- **FIELD**: SET_FIELD, GET_FIELD, GET_USERS_FIELDS
- **PARAM**: LIST_USERS_PARAMS, SET_PARAM, DEL_PARAM, LINK_FIELD_PARAM, RM_LINK_FIELD_PARAM, GET_FIELDS_PARAMS
- **MODULE**: SET_MODULE
- **METHOD**: SET_METHOD
- **FILE**: SET_FILE
- **SESSION**: SET_SESSION
- **INJECTION**: SET_INJECTION
- **CONTROL**: SPAWN_OBJECT, GET_AVAILABLE_OBJECTS
- **RESEARCH**: COLLECT_INFORMATION, ANALYZE_SIM_RESULTS, START_RESEARCH

---

## Cursor IDE

### MCP

- **cursor-ide-browser**: Navigate, snapshots, click, type, fill forms. Use for frontend testing.
- **MCP descriptors**: `mcps/cursor-ide-browser/tools/`

### Skills

- `create-rule` – Cursor rules
- `create-skill` – Agent skills
- `update-cursor-settings` – settings.json

---

## OpenAI Apps SDK

Publish BestBrain as a ChatGPT app.

- **MCP server**: `py -m _admin.app_handler.openai_asdk.mcp_server --port 8787`
- **Docker**: `docker build -t bestbrain-mcp-app -f _admin/app_handler/openai_asdk/Dockerfile.mcp .`
- **Connector URL**: `https://<your-domain>/mcp`
- **Docs**: `_admin/app_handler/openai_asdk/README.md`

---

## QDash Frontend

- **Gemini**: Set `CLIENT_KEY_GEMINI_API_KEY` for AI terminal
- **WebSocket**: Connects to backend relay
- **Run**: `cd qdash && npm start`

---

## Researcher2

Deep research agent (Gemini or ChatGPT).

- **Backend**: `DEEP_RESEARCH_BACKEND=gemini` or `chatgpt`
- **Gemini**: Set `GEMINI_API_KEY`
- **Run**: `py -m qbrain.core.researcher2.researcher2.cli --prompt "Your prompt"` (or `researcher2` if installed)

---

## Admin CLI

```bash
py -m _admin.main --scan-only              # List projects
py -m _admin.main --run-local              # Start backend + frontend
py -m _admin.main --write-gemini-settings   # Write Gemini settings.json
py -m _admin.main --record-qdash-demo       # Record qdash demo (MP4 + HTML)
py -m _admin.main --publish-app            # MCP Docker + health + checklist
py -m _admin.main --deploy                  # Build + deploy to K8s
```

---

## Guard (Simulation)

```bash
py -m qbrain.core.guard
```

- Uses DuckDB, JAX, grid workflow
- Env: `GRID_CFG_PATH`, `GRID_CMD`, `GRID_MODEL_OUT`

---

## Environment Variables Summary

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini API (QDash, Researcher2, mcpmaster) |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | GitHub MCP (Gemini CLI) |
| `RELAY_WS_URL` | QBrain WebSocket (optional) |
| `GCP_PROJECT_ID`, `GCP_REGION` | GCP / Vertex AI |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP auth |
| `DJANGO_SETTINGS_MODULE` | `qbrain.bm.settings` |
| `OPENAI_API_KEY` | Researcher2 (chatgpt backend) |
| `DEEP_RESEARCH_BACKEND` | `gemini` or `chatgpt` |

---

## Project Layout

```
BestBrain/
├── qbrain/          # Backend (Django, relay, core, _db)
├── qdash/            # React frontend
├── _admin/           # Admin CLI, deploy, MCP
├── qbrain/mcpmaster/ # Gemini extension + relay cases
├── .gemini/          # Gemini CLI config
└── .env              # Secrets (gitignored)
```

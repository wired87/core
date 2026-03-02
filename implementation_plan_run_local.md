# Implementation plan: run entire project locally

Execute these steps in order to run the full stack (backend + frontend) on your machine.

---

## Prerequisites

- **Python** (py / python) with project dependencies installed
- **Node.js** and **npm** for the qdash frontend
- Ports **8000** (backend) and **3000** (frontend) free

---

## Plan (steps)

### Step 1: Set environment (optional but recommended)

Skip slow DB table init on first backend start:

```powershell
$env:TABLE_EXISTS = "True"
```

(Optional: set `PYTHONPATH` and `DJANGO_SETTINGS_MODULE` if not already set; `run_daphne.py` uses the project root.)

### Step 2: Start backend (Django + Daphne, WebSocket)

From **project root** (BestBrain):

```powershell
py qbrain\run_daphne.py
```

On Windows PowerShell use one line per command or `;` to separate (avoid `&&` in older PowerShell).

- Leave this terminal open. Backend is ready when you see the server listening (e.g. on 8000).
- Health: http://127.0.0.1:8000/health/
- WebSocket: ws://127.0.0.1:8000/run/

### Step 3: Start frontend (React qdash)

In a **second terminal**, from project root:

```powershell
cd qdash
npm start
```

- App: http://127.0.0.1:3000/
- Proxy in `qdash/package.json` forwards API calls to backend (8000).

### Step 4: Verify

- Open http://127.0.0.1:3000 in a browser (first load may take a few seconds).
- Backend health: http://127.0.0.1:8000/health/ → `{"status":"ok","db":"connected"}`.

---

## Alternative: single command (admin)

From project root, run all discovered apps (backend + frontend) in one go:

```powershell
$env:TABLE_EXISTS = "True"
py -m _admin.main --run-local-testing
```

- Starts backend (Daphne) and frontend (qdash) as subprocesses; Ctrl+C stops both.
- `--run-local-testing`: backend + React web only, sets TESTING=1.

Or with **separate terminal windows** (Windows: each app in its own console):

- The admin CLI does not expose a “separate terminals” flag in the main parser; use the two-terminal flow above for that.

---

## Summary

| Step | Action | Terminal |
|------|--------|----------|
| 1 | `$env:TABLE_EXISTS = "True"` (optional) | Any |
| 2 | `py qbrain\run_daphne.py` | 1 (keep open) |
| 3 | `cd qdash` then `npm start` | 2 (keep open) |
| 4 | Open http://127.0.0.1:3000 | Browser |

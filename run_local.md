# Run frontend + backend locally

## Backend (Django + Daphne, port 8000)

From **project root** (BestBrain):

```powershell
# Optional: skip slow DB table init on first run (tables created on first use)
$env:TABLE_EXISTS = "True"

# Start ASGI server (HTTP + WebSocket)
py qbrain\run_daphne.py
```

Or from **qbrain** directory:

```powershell
$env:TABLE_EXISTS = "True"
cd c:\Users\bestb\PycharmProjects\BestBrain
py -c "import sys; sys.path.insert(0, '.'); from pathlib import Path; sys.path.insert(0, str(Path('.').resolve().parent)); exec(open('run_daphne.py').read())"
```

Simpler: from project root, `py qbrain\run_daphne.py` (with PYTHONPATH set by the script).

- Health: http://127.0.0.1:8000/health/
- WebSocket: ws://127.0.0.1:8000/run/?user_id=...&mode=demo

## Frontend (React qdash, port 3000)

From **project root**:

```powershell
cd qdash
npm start
```

- App: http://127.0.0.1:3000/
- In development, the frontend uses `ws://127.0.0.1:8000/run/` for the relay and `package.json` has `"proxy": "http://127.0.0.1:8000"` so API calls to the same origin are forwarded to the backend.

## Order

1. Start **backend** first (Daphne takes ~20–30 s to listen).
2. Then start **frontend** (`npm start` in qdash).

## Troubleshooting

- **Backend won’t start**: Ensure port 8000 is free. Use `TABLE_EXISTS=True` to skip initial table creation.
- **WebSocket disconnect**: Ensure Daphne is running (not `manage.py runserver`, which is WSGI-only).
- **Frontend “Something is already running on port 3000”**: Either use that tab or stop the process using 3000 and run `npm start` again.

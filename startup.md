# Startup: frontend + backend

## 1) Backend (leave this running)

```powershell
$env:TABLE_EXISTS = "True"
py qbrain\run_daphne.py
```

## 2) In another terminal: frontend

```powershell
cd qdash
npm start
```

Then open **http://127.0.0.1:3000**.

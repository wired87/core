# Run QBrain: activate .venv, set PYTHONPATH, start Daphne (bm.asgi).
# Usage: from qbrain dir: .\run.ps1   or from repo root: .\qbrain\run.ps1
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $ScriptDir
$env:PYTHONPATH = "$RepoRoot$([IO.Path]::PathSeparator)$ScriptDir"
$env:DJANGO_SETTINGS_MODULE = "qbrain.bm.settings"

$venv = Join-Path $ScriptDir ".venv"
if (Test-Path (Join-Path $venv "Scripts\Activate.ps1")) {
    & "$venv\Scripts\Activate.ps1"
}
python -m daphne -b 0.0.0.0 -p 8080 bm.asgi:application

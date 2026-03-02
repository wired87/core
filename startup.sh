#!/bin/bash
# Start all applications (React, DRF) discovered by admin, each in its own terminal.
# Run from project root: ./startup.sh   or   bash startup.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

# Final command: admin starts backend (Django/DRF) and frontend (React) in separate terminals
exec python3 -m _admin.main --local --separate-terminals

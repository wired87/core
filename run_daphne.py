#!/usr/bin/env python
"""
Run Daphne ASGI server with WebSocket-friendly timeouts.

Use: python run_daphne.py

Fixes "Application instance took too long to shut down and was killed" by:
1. --application-close-timeout 30: allows graceful connection cleanup
2. Relay.connect() runs OrchestratorManager creation in sync_to_async (thread pool)
   so sync BigQuery/init code does not block the event loop

Always use this script for WebSocket/relay; avoid: daphne bm.asgi:application
without the timeout flag.
"""
import os
import subprocess
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bm.settings")

if __name__ == "__main__":
    port = os.environ.get("PORT", "8000")
    cmd = [
        sys.executable, "-m", "daphne",
        "--application-close-timeout", "30",
        "-b", "0.0.0.0",
        "-p", port,
        "bm.asgi:application",
    ]
    sys.exit(subprocess.run(cmd).returncode)

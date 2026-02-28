#!/bin/sh
# Production entrypoint: run uvicorn with PORT from env (default 8000).
set -e
port="${PORT:-8000}"
exec uvicorn qbrain.bm.asgi:application --host 0.0.0.0 --port "$port"

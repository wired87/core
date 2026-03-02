#!/bin/bash

# Project root (parent of qbrain/) so qbrain is importable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT_ROOT"
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-qbrain.bm.settings}"

# 1. Render Nginx config from env (e.g. Domain)
python3 -m qbrain.nginx.render_nginx_conf

# 2. Copy generated config to Nginx site
# Script writes e.g. localhost.conf into ./nginx/
cp "$SCRIPT_DIR/nginx/"*.conf /etc/nginx/sites-enabled/default

# 3. Start Nginx in background
nginx -g "daemon on;"

# 4. Start the app (Daphne ASGI)
PORT="${PORT:-8080}"
exec python3 -m daphne --application-close-timeout 30 -b 0.0.0.0 -p "$PORT" qbrain.bm.asgi:application
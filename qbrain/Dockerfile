# qbrain engine: Django + Channels ASGI served with uvicorn (production).
# Build from project root: docker build -f qbrain/Dockerfile -t qbrain-engine .
# Run: docker run -p 8000:8000 -e DJANGO_SETTINGS_MODULE=qbrain.bm.settings qbrain-engine

FROM python:3.10-slim AS builder

WORKDIR /build

# System deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install deps (use project root r.txt)
COPY r.txt .
RUN pip install --no-cache-dir --user -r r.txt

# --- Runtime stage ---
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy only what the app needs: project root layout so BASE_DIR and qbrain package resolve
COPY manage.py .
COPY qbrain ./qbrain

# Mount .env at run if needed: docker run -v $(pwd)/.env:/app/.env ...

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=qbrain.bm.settings \
    PORT=8000

EXPOSE 8000

# Production: uvicorn serves Django/Channels ASGI; single worker for WebSockets
# PORT overridable at run: docker run -e PORT=8080 -p 8080:8080 ...
COPY qbrain/docker_entry.sh /docker_entry.sh
RUN chmod +x /docker_entry.sh
ENTRYPOINT ["/docker_entry.sh"]

# --- Build Stage ---
FROM python:3.10-slim AS builder

WORKDIR /usr/src/app

# System-Abhängigkeiten für Build-Prozesse
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cache-Optimierung für Pip
COPY r.txt .
RUN pip install --user --no-cache-dir -r r.txt

# --- qdash Frontend Build Stage ---
FROM node:20-slim AS qdash-builder

WORKDIR /qdash
COPY qdash/package*.json ./
RUN npm ci 2>/dev/null || npm install
COPY qdash .
RUN npm run build

# --- Final Stage ---
FROM python:3.10-slim AS runner

WORKDIR /usr/src/app

# Kopiere installierte Pakete vom Builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Installiere Nginx (falls du beides in einem Container/Pod nutzt)
# Hinweis: In Cloud-Umgebungen läuft Nginx oft separat.
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

# Projekt-Dateien kopieren
COPY . .

# Pre-built qdash frontend (startup.py --skip-frontend copies to static_root at runtime)
COPY --from=qdash-builder /qdash/build ./qdash/build

# Umgebungsvariablen für JAX/Django/Nginx
ENV PYTHONPATH=/usr/src/app \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    NGINX_APP_PORT=8080 \
    NGINX_USE_HTTP_ONLY=true \
    NGINX_STATIC_ROOT=/usr/src/app/static_root \
    NGINX_MEDIA_ROOT=/usr/src/app/media \
    DJANGO_SETTINGS_MODULE=qbrain.bm.settings

# 🔹 Rendering der Nginx-Config während des Builds (HTTP-only für Container)
RUN python3 -m qbrain.nginx.render_nginx_conf

# Port 80 für Nginx und 8080 für Daphne
EXPOSE 80 8080

# Unified startup: migrate, collectstatic, copy qdash, nginx, Daphne
CMD ["python", "startup.py", "--skip-frontend"]
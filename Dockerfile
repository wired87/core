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

# Umgebungsvariablen für JAX/Django/Nginx
ENV PYTHONPATH=/usr/src/app \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    NGINX_APP_PORT=8080 \
    DJANGO_SETTINGS_MODULE=bm.settings

# 🔹 Rendering der Nginx-Config während des Builds (für Default-Werte)
# Dies nutzt dein render_nginx_conf.py Skript
RUN python3 -m nginx.render_nginx_conf

# Startup-Skript ausführbar machen
RUN chmod +x startup.sh

# Port 80 für Nginx und 8080 für Daphne
EXPOSE 80 8080

# Nutze das Startup-Skript, um Env-Variablen zur Laufzeit in Nginx zu injizieren
CMD ["/bin/bash", "./startup.sh"]
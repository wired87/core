#!/bin/bash

# 1. Rendere die Nginx Config basierend auf aktuellen Env-Vars (z.B. Domain)
python3 -m nginx.render_nginx_conf

# 2. Verschiebe die generierte Datei an den Nginx Ort
# Das Skript generiert Dateien wie 'localhost.conf' in ./nginx/
cp ./nginx/*.conf /etc/nginx/sites-enabled/default

# 3. Starte Nginx im Hintergrund
nginx -g "daemon on;"

# 4. Starte die App (Korrektur des Tippfehlers: application)
exec daphne -b 0.0.0.0 -p 8080 bm.asgi:application
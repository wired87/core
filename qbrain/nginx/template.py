NGINX_TEMPLATE="""
server {
    # HTTP -> HTTPS redirect
    listen 80;
    server_name <DOMAIN> www.<DOMAIN>;

    return 301 https://$host$request_uri;
}

server {
    # Main HTTPS server
    listen 443 ssl http2;
    server_name <DOMAIN> www.<DOMAIN>;

    # TLS configuration (paths managed by Certbot)
    ssl_certificate /etc/letsencrypt/live/<DOMAIN>/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/<DOMAIN>/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    # --- Application proxy (Django/FastAPI/ASGI on localhost) ---
    location / {
        proxy_pass http://127.0.0.1:<APP_PORT>;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts for long-running requests
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        send_timeout 600s;
    }

    # --- Static files ---
    location /static/ {
        alias <STATIC_ROOT>/;
    }

    # --- Media files ---
    location /media/ {
        alias <MEDIA_ROOT>/;
    }

    # --- Additional API prefix (example: /inv/) ---
    location /inv/ {
        proxy_pass http://127.0.0.1:<APP_PORT>;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # --- WebSocket endpoint (example: /qf-sim/) ---
    location /qf-sim/ {
        proxy_pass http://127.0.0.1:<APP_PORT>;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
"""

# HTTP-only template for container use (e.g. Cloud Run, TLS at load balancer)
NGINX_HTTP_ONLY_TEMPLATE = """
server {
    listen 80;
    server_name <DOMAIN> _;

    # --- Application proxy (Django/ASGI) ---
    location / {
        proxy_pass http://127.0.0.1:<APP_PORT>;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        send_timeout 600s;
    }

    # --- WebSocket /run ---
    location /run {
        proxy_pass http://127.0.0.1:<APP_PORT>;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # --- Static files ---
    location /static/ {
        alias <STATIC_ROOT>/;
    }

    # --- Media files ---
    location /media/ {
        alias <MEDIA_ROOT>/;
    }
}
"""
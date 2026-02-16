### BestBrain Nginx setup

This folder captures the Nginx configuration and commands needed to expose the BestBrain application on a custom domain, based on `utils/setup/nginx/setup.txt`.

The config is now **rendered from the app's `.env`** by `nginx/render_nginx_conf.py`, so you can control the public domain and app port via environment variables instead of editing the config by hand.

#### 1. Environment variables used for rendering

`nginx/render_nginx_conf.py` loads `.env` from the project root (if present) and resolves values in this order:

- **Domain** (`<DOMAIN>` in the template):
  - `NGINX_DOMAIN` → `DOMAIN` → `CLUSTER_DOMAIN` → fallback `localhost`
- **App port** (`<APP_PORT>` in the template):
  - `NGINX_APP_PORT` → `PORT` → `CLUSTER_PORT` → fallback `8080`
- **Static and media roots**:
  - `NGINX_STATIC_ROOT` (default `/var/www/bestbrain/static_root`)
  - `NGINX_MEDIA_ROOT` (default `/var/www/bestbrain/media`)

When you run:

```bash
python -m nginx.render_nginx_conf
```

the script:

1. Loads `.env` from the project root (if it exists).
2. Renders the template from `nginx/template.py` with the resolved values.
3. Writes a config file under `nginx/`, named `<domain>.conf` (domain-based).

This makes the Nginx layer modular: changing `.env` and re‑running the script regenerates a matching config.

#### 2. Install Nginx and Certbot (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
sudo systemctl enable nginx
sudo systemctl start nginx
sudo systemctl status nginx
```

#### 3. Configure the site (host install)

If you are using `startup.sh --init` on a host, the script will:

1. Run `python -m nginx.render_nginx_conf` after Django migrations and `collectstatic`.
2. If `/etc/nginx/sites-available/bm` does not exist yet, copy the rendered config (first `nginx/*.conf`) into `/etc/nginx/sites-available/bm`.
3. Symlink `/etc/nginx/sites-enabled/bm` → `/etc/nginx/sites-available/bm` and reload Nginx.

If you prefer to manage this manually:

1. Render the config:

   ```bash
   python -m nginx.render_nginx_conf
   ls nginx/*.conf  # pick the generated file, e.g. nginx/bestbrain_conf.com
   ```

2. Copy it to `/etc/nginx/sites-available` and enable it:

   ```bash
   sudo cp nginx/<your-domain>.conf /etc/nginx/sites-available/bestbrain.conf
   sudo ln -s /etc/nginx/sites-available/bestbrain.conf /etc/nginx/sites-enabled/bestbrain.conf
   ```

#### 4. Test and reload Nginx

```bash
sudo nginx -t
sudo systemctl reload nginx
```

#### 5. Obtain/renew TLS certificates with Certbot

Run Certbot with the Nginx plugin (will inject SSL directives into the server block):

```bash
sudo certbot --nginx -d bestbrain.tech -d www.bestbrain.tech
```

Adjust the domains to match your `<DOMAIN>` and `www.<DOMAIN>`.

#### 6. Timeouts (long‑running requests)

To increase timeouts for long requests, the example config already includes:

- `proxy_read_timeout 600s;`
- `proxy_connect_timeout 600s;`
- `proxy_send_timeout 600s;`
- `send_timeout 600s;`

You can move these into the `http` block in `/etc/nginx/nginx.conf` if you want them to apply globally.

#### 7. WebSockets

The example config includes a `/qf-sim/` location pre‑configured for WebSockets:

- `proxy_http_version 1.1;`
- `proxy_set_header Upgrade $http_upgrade;`
- `proxy_set_header Connection "upgrade";`

Point the location to your WebSocket server (e.g. Daphne/Uvicorn on `<APP_PORT>`).

#### 8. Reload sequence after deployments

After pulling new code or restarting backend services:

```bash
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl restart gunicorn    # or your ASGI/WSGI server
sudo systemctl restart fail2ban    # if used
```


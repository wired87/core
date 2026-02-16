"""
Render a concrete Nginx config from the embedded template using environment variables.

Usage (inside project root or container):

    python -m nginx.render_nginx_conf

Environment variables (with precedence):

- NGINX_DOMAIN → DOMAIN → CLUSTER_DOMAIN     -> replaces <DOMAIN>
- NGINX_APP_PORT → PORT → CLUSTER_PORT      -> replaces <APP_PORT>
- NGINX_STATIC_ROOT                         -> replaces <STATIC_ROOT>
- NGINX_MEDIA_ROOT                          -> replaces <MEDIA_ROOT>
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from nginx.template import NGINX_TEMPLATE


def _load_env_from_project_root() -> None:
    """
    Load .env from the project root (parent of nginx/), if present.
    This allows the render script to use the same env config as the app.
    """
    try:
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except Exception as e:
        # Non-fatal: if .env cannot be loaded we simply rely on process env.
        print(f"[nginx.render_nginx_conf] Could not load .env: {e}")


def _resolve_domain() -> str:
    """
    Resolve the public domain in a way that aligns with the app's .env:

    NGINX_DOMAIN -> DOMAIN -> CLUSTER_DOMAIN -> 'localhost'
    """
    domain = (
        os.environ.get("NGINX_DOMAIN")
        or os.environ.get("DOMAIN")
        or os.environ.get("CLUSTER_DOMAIN")
        or "localhost"
    )
    return domain.strip()


def _resolve_app_port() -> str:
    """
    Resolve the backend app port:

    NGINX_APP_PORT -> PORT -> CLUSTER_PORT -> '8080'
    """
    app_port = (
        os.environ.get("NGINX_APP_PORT")
        or os.environ.get("PORT")
        or os.environ.get("CLUSTER_PORT")
        or "8080"
    )
    return str(app_port).strip()


def render() -> Path:
    """
    Render the Nginx configuration from the template using env variables.
    Returns the path to the generated config file under nginx/.
    """
    base_dir = Path(__file__).resolve().parent

    # Ensure .env from project root is loaded (if present)
    _load_env_from_project_root()

    domain = _resolve_domain()
    app_port = _resolve_app_port()

    static_root = os.environ.get("NGINX_STATIC_ROOT") or "/var/www/bestbrain/static_root"
    media_root = os.environ.get("NGINX_MEDIA_ROOT") or "/var/www/bestbrain/media"

    # Normalize roots (avoid double slashes)
    static_root = static_root.rstrip("/")
    media_root = media_root.rstrip("/")

    # Build concrete config from template
    rendered = (
        NGINX_TEMPLATE
        .replace("<DOMAIN>", domain)
        .replace("<APP_PORT>", str(app_port))
        .replace("<STATIC_ROOT>", static_root)
        .replace("<MEDIA_ROOT>", media_root)
    )

    # Use a deterministic, domain-based filename (dots replaced for safety)
    domain_safe = (domain or "default").replace("://", "_").replace("/", "_")
    domain_safe = domain_safe.replace("*", "star").replace("..", ".")
    output_name = f"{domain_safe}.conf"
    output_path = base_dir / output_name

    output_path.write_text(rendered, encoding="utf-8")
    print(f"[nginx.render_nginx_conf] Wrote config to {output_path}")
    return output_path


if __name__ == "__main__":
    render()


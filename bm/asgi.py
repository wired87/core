"""
ASGI config for bm project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import os

from bm import routing

if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
    if os.name == "nt":
        GOOGLE_APPLICATION_CREDENTIALS = r"C:\\Users\\wired\\OneDrive\\Desktop\\Projects\\bm\utils\ggoogle\\g_auth\\aixr-401704-59fb7f12485c.json"
    else:
        GOOGLE_APPLICATION_CREDENTIALS = "/home/derbenedikt_sterra/bm/utils/ggoogle/g_auth/aixr-401704-59fb7f12485c.json"

    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GOOGLE_APPLICATION_CREDENTIALS)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bm.settings')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    # Just HTTP for now. (We can add other protocols later.)
    "websocket": AuthMiddlewareStack( # Wenden Sie optional AuthMiddlewareStack an
            URLRouter(
                # Binden Sie Ihre App-spezifischen WebSocket-Routen hier ein
                routing.websocket_urlpatterns
            )
        ),
})




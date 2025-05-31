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
        FIREBASE_CREDENTIALS = r"C:\\Users\\wired\OneDrive\\Desktop\\Projects\\Brainmaster\\_google\\g_auth\\firebase_creds.json"

    else:
        GOOGLE_APPLICATION_CREDENTIALS = "/home/derbenedikt_sterra/bm/utils/ggoogle/g_auth/aixr-401704-59fb7f12485c.json"
        FIREBASE_CREDENTIALS = r"_google/g_auth/firebase_creds.json"
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GOOGLE_APPLICATION_CREDENTIALS)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bm.settings')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
            URLRouter(
                routing.websocket_urlpatterns
            )
        ),
    }
)




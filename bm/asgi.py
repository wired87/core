"""
ASGI config for bm project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""






import sys
import numpy as np
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    from unittest.mock import MagicMock
    mock_jax = MagicMock()
    mock_jax.numpy = np
    sys.modules["jax"] = mock_jax
    sys.modules["jax.numpy"] = np
    jnp = np


from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import os

from auth.set_gcp_auth_creds_path import set_gcp_auth_path
from bm import routing

set_gcp_auth_path()
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




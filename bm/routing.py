from django.urls import re_path
from relay_station import Relay

websocket_urlpatterns = [
    re_path(r"^run/?$", Relay.as_asgi()),
]



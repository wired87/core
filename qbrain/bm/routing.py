from django.urls import re_path
from qbrain.relay_station import Relay

websocket_urlpatterns = [
    re_path(r"^run/?$", Relay.as_asgi()),
]



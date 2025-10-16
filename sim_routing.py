

from django.urls import re_path

from relay_station import Relay

sim_ws_urls = [
    re_path(r"^run/?$", Relay.as_asgi()),
]



# your_app/routing.py
from qf_sim.dj.sim_routing import sim_ws_urls

websocket_urlpatterns = [
    *sim_ws_urls,
]



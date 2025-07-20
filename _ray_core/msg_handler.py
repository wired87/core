import asyncio

import ray


@ray.remote
class MessageHandler:

    """
    Add on WS relay_station

    For
    - sim updates (stop, pause etc)
    - database changes listener
    - distributor
    """

    def __init__(self, g):
        self.g = g

        # Get DB Paths for all Sub fields (each _ray_core worker == single sub field)
        self.db_listener_paths = [
            f"{self.g.database}/{attrs.get('type')}/{nid}" for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") in ALL_SUBS + "ENV"
        ]

        self.loop = asyncio.get_event_loop




    def _handle_incoming_data(self, data):
        attrs = data["data"]
        path = data["path"]
        if path.endswith("/"):
            path = path[:-1]
        nid = path.split("/")[-1]





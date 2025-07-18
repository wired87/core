import os
import ray
import socket
import networkx as nx
from ray import serve

# === Step 1: IP und Umgebungsvariablen laden ===
ip = socket.gethostbyname(socket.gethostname())
port = os.environ.get("RAY_PORT", "6379")
head_address = f"{ip}:{port}"


# === Step 2: Ray starten ===
ray.init(address=f"ray://{head_address}")

# === Step 3: Serve starten und verbinden ===
serve.start(detached=True)

# === Step 4: Graph laden ===
with open(graph_path, "rb") as f:
    G: nx.Graph = nx.node_link_data(f)

# === Step 5: Remotes erzeugen und beim Head registrieren ===
for nid, attrs in G.nodes(data=True):
    ref = WorkerNode.remote(G, attrs, env_id=env_id)
    ray.get(ref.register_to_head.remote())  # optional: Registrierung beim Head

print("Startup abgeschlossen.")

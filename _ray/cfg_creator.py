from utils.graph.local_graph_utils import GUtils


class CfgCreator:


    def __init__(self, g:GUtils):
        self.g = g



    def main(self):
        self._spec_head()
        return



    def _spec_head(self):
        """
        → Er ist der zentrale Controller deines Clusters (Scheduler + Object Store + gRPC/Serve/API).
        → Kubernetes-Beschreibung für den einzigen Pod, der den Cluster verwaltet.
        Er startet Ray mit --head, ist kein Worker, aber du kannst in ihm:

        ray.init() ausführen

        ray.remote Klassen erstellen

        ray.serve starten ✅

        Kann universell sein
        """
        self.head_spec = f"""
        headGroupSpec:
        rayStartParams:
            dashboard-host: "0.0.0.0"
        template:
            spec:
              containers:
              - name: ray-head
                image: dein/image:mit/main.py
                command: ["python", "main.py"]
        """
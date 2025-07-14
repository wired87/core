import ray


@ray.remote
class VisualizerWorker:

    """
    Erhält einzelnen qfn zum update -> todo geplantes sys

    ToDo:
    DB upsert prozesse auf sepparates dbworker _qfn_cluster_node ausweiten
    """

    def __init__(
            self,
            g,
            attrs,
            env,
            nid,
            parent,
    ):
        self.id=nid
        self.attrs=attrs


    def main(self):
        return
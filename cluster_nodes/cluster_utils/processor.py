"""
Available for all FieldWorkers
Takes updates, processes them for all kinds of actions (Visual, ML, DB)
and distributes them to the specific services
"""

import ray.data

from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from cluster_nodes.manager.trainer import LiveTrainer
from cluster_nodes.manager.visualizer import VisualizerWorker


@ray.remote
class DataProcessorWorker:
    def __init__(
            self,
            visualizer,
            trainer,
    ):
        self.node_type = "data_processor"
        self.visualizer=visualizer
        self.trainer=LiveTrainer.remote()

        self.visual_worker_ref = VisualizerWorker.remote()
        self.receiver = ReceiverWorker.remote(
            ntype=self.node_type
        )

    async def main(self, attrs):
        """Gets called from receiver"""
        await self._convert_ml(attrs)
        await self.convert_visual(attrs)
        await self.convert_db(attrs)

    async def convert_ml(self):
        return


    async def convert_visual(self):
        return


    async def convert_db(self):
        return


    async def redirect(self):
        return
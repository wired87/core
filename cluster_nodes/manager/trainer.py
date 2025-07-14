import asyncio

import ray
import torch
import torch.nn as nn
import torch.optim as optim

from cluster_nodes.cluster_utils.receiver import ReceiverWorker
from utils.graph.local_graph_utils import GUtils


@ray.remote
class LiveTrainer:

    """
    Live data trainer auf qfn basis
    """

    def __init__(self, G, user_id):
        self.node_type = "trainer"
        self.G = G
        self.queue = asyncio.Queue()
        self.model = nn.Linear(1, 1)
        self.opt = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.user_id=user_id
        self.batch_size = 16
        self.loop_task = asyncio.create_task(self._train_forever())

        self.receiver = ReceiverWorker.remote(
            ntype=self.node_type
        )

        self.g = GUtils(
            nx_only=False,
            G=self.G,
            g_from_path=None,
            user_id=self.user_id,
        )  # -> DataManager -> local

        print("[LiveTrainer] Initialisiert.")

    async def add_sample(self, x, y):
        await self.queue.put((x, y))  # Worker ruft das pro Schritt auf

    async def _train_forever(self):
        print("[LiveTrainer] Starte Trainingsloop...")
        while True:
            batch = []
            while len(batch) < self.batch_size:
                batch.append(await self.queue.get())

            # → Torch Dataset
            x = torch.tensor([v[0] for v in batch]).unsqueeze(1)
            y = torch.tensor([v[1] for v in batch]).unsqueeze(1)

            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            print(f"[TRAIN] Batch done. Loss: {loss.item():.4f}")

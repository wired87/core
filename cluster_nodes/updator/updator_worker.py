import asyncio
import time

import ray

from cluster_nodes.cluster_utils.listener import Listener
from cluster_nodes.cluster_utils.processor import DataProcessorWorker
from qf_core_base.calculator.calculator import Calculator
from qf_core_base.qf_utils.all_subs import ALL_SUBS
from qf_core_base.runner.qf_updator import QFUpdator
from utils.logger import LOGGER

class NeighborHandler:

    def __init__(self, neighbors, host):
        self.neighbors = neighbors
        self.host = host
        # Listens to live state changes to distribute
        self.neighbor_change_listener = Listener.remote(
            paths_to_listen=[
                ray.get(self.host["db_worker"].firebase.get_listener_endpoints.remote(
                    nodes=list(self.neighbors.keys()),
                ))
            ],
            db_manager=ray.get(self.host["db_worker"].get_db_manager.remote()),
            host=self.host,
            listener_type="db_neighbor_changes",
        )

    async def apply_db_neighbor_changes(self, nid, attrs):
        LOGGER.info(f"Update attrs for n: {nid}")
        if nid in self.neighbors:
            self.neighbors[nid].update(attrs)
        else:
            LOGGER.warning(f"Neighbor {nid} not found in current neighbors list")

    async def _send_changes_to_neighbors(self):
        # todo when changes are detected,
        #  send to DB Worker Node
        # send updated attrs to neighbors
        tasks = [
            nattrs["ref"].receiver.receive.remote(
                {
                    "type": "n_change",
                    "data": self.attrs
                })
            for nnid, nattrs in self.neighbor_handler.neighbors
        ]
        await asyncio.gather(*tasks)



@ray.remote
class UpdatorWorker:

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
            parent:str,
            host,
            neighbor_struct
    ):
        self.id=nid
        self.attrs=attrs
        self.parent=parent
        self.run=False
        self.env=env
        self.prev = self.attrs[self.parent.lower()]
        self.g=g
        self.host=host

        self.updator = QFUpdator(
            g,
            env,
            testing = False,
            specs = {},
            run=self.run
        )
        self.qfn_parent_id, self.qfn_parent_attrs = self.g.get_single_neighbor_nx(self.id, "QFN")

        self.calculator = Calculator(g)


        # Niehbor handler
        self.neighbor_handler = NeighborHandler(
            neighbors=neighbor_struct,
            host=self.host
        )

        # set pm id -> get attrs in each iter
        self.neighbors_pm = self.calculator.cutils.set_neighors_plus_minus(
            node_id=self.qfn_parent_id,
            self_attrs=self.qfn_parent_attrs,
            d=self.env["d_default"],
            trgt_type="QFN",
            field_type=self.parent.lower()
        )

        # AI and visual Data processor
        self.processor = DataProcessorWorker.remote()

        # Equations todo -> gleichungen direkt übern graphen laden
        self.arsenal = self.calculator.cutils.arsenal[
            self.parent.lower()
        ]
        self.type=type
        self.g.G=None
       #print("main updator intialized")


    async def main(self, start_time):
        if start_time >= int(time.time()):
            await asyncio.sleep(.1)

        await self.update()


    async def update(self):

        """
        Updated sich slebst. Du startest ihn von außen und ist dann völlig autonom
        du kannst commands unn rt einbringen
        circuits ins gehirn injizieren -> updates wo immer du bist ( lerne zB temporär sprachen, erweitere dein wissen in die cloud und hol dir das was du brauchst situationsspezifisch.)
        """
        while self.run is True:
            # Node scans each iter for neighbors in env
            self.updator.update_core(
                nid=self.id,
                attrs=self.attrs,
                qf_attrs=self.qfn_parent_attrs,
                qf_nid=self.qfn_parent_id,
                neighbors_pm=self.neighbors_pm,
                env_attrs=self.env,
            )

            #changed = self._validate_changes(self.attrs)

            await self.send_process_data()
            print(f"finished update {self.id}")
            self.attrs["time"] += self.env["timestep"]
            # todo check len history -> load queue -> upsert
            ray.get(self.host["db_worker"].iter_upsert.remote(self.attrs))
        else:
            LOGGER.info(f"Update for {self.id} finished")
            # todo shutdown


    async def send_process_data(self):
        await self.processor.receiver.receive.remote(
            payload={
                "data": self.attrs,
                "type": "data_update"
            }
        )







    def stop(self):
        self.run = False
       #print("sim stopped")

    def start(self):
        self.run = True
       #print("sim started")


    def _validate_changes(self, updated_attrs):
       #print("validate changes")
        changed = False
        for k, v in updated_attrs.items():
            if self.attrs[k] != v:
                changed = True
                self.attrs.update(updated_attrs)
                break
        return changed

import asyncio
import time

import ray

from cluster_nodes.cluster_utils.processor import DataProcessorWorker
from qf_core_base.calculator.calculator import Calculator
from qf_core_base.qf_utils.all_subs import ALL_SUBS
from qf_core_base.runner.qf_updator import QFUpdator
from utils.logger import LOGGER


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
            parent,
            db_worker
    ):
        self.id=nid
        self.attrs=attrs
        self.parent=parent
        self.run=False
        self.env=env
        self.prev = self.attrs[self.parent.lower()]
        self.db_worker=db_worker
        self.g=g

        self.updator = QFUpdator(
            g,
            env,
            testing = False,
            specs = {},
            run=self.run
        )
        self.qfn_parent_id, self.qfn_parent_attrs = self.g.get_single_neighbor_nx(self.id, "QFN")

        self.calculator = Calculator(g)

        self.neighbors = self.g.get_neighbor_list(
            self.id,
            target_type=ALL_SUBS
        )

        # set pm id -> get attrs in each iter
        self.neighbors_pm = self.calculator.cutils.set_neighors_plus_minus(
            node_id=self.qfn_parent_id,
            self_attrs=self.qfn_parent_attrs,
            d=self.env["d_default"],
            trgt_type="QFN",
            field_type=self.parent.lower()
        )

        self.processor = DataProcessorWorker.remote()

        # Equations todo -> gleichungen direkt übern graphen laden
        self.arsenal = self.calculator.cutils.arsenal[
            self.parent.lower()
        ]
        self.type=type
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

            await self._send_changes_to_neighbors()
            await self.send_process_data()
            print(f"finished update {self.id}")
            self.attrs["time"] += self.env["timestep"]
            # todo check len history -> load queue -> upsert
            self.db_worker.iter_upsert(self.attrs)
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
            for nnid, nattrs in self.neighbors
        ]
        await asyncio.gather(*tasks)


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

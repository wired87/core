import asyncio
import os

import ray
from ray import serve, get_actor
from ray.serve.handle import DeploymentHandle

from a_b_c._creators.utils import CloudRcsCreator
from a_b_c.bq_agent.bq_worker import BQService
from a_b_c.gemw.gem import Gem
from a_b_c.spanner_agent._spanner_graph.emulator import SpannerEmulatorManager
from a_b_c.spanner_agent.spanner_agent import SpannerWorker
from core.app_utils import TABLE_NAME, DOMAIN, ENV_ID
from _ray_core.base.base import BaseActor
from a_b_c.db_worker import FBRTDBAdminWorker
from _ray_core.globacs.state_handler.main import StateHandler
from qf_utils.all_subs import ALL_SUBS
from utils.utils import Utils


@ray.remote
class CloudMaster(
    Utils,
    BaseActor,
    StateHandler
):
    """
    todo create one universal creator worker and cloudupdaor workers -> connect ot cli (+docs)) e.g.: gem worker holds vector store (vs) with fetched docs (or uses google api to fetch -> process (like convert to G format) -> embed)
    Creates Core rcs:
    BQ
    - DB : ENV ID
    - Tables for time series-> create for each nid a table to avoid persistency issues
    SP
    - instance
    - db
    - tables (just for states -> classified in ntypes)

    AND
    uses Relay to create admin_data

    Finals state = All daa inside all resourcess
    """

    def __init__(
            self,
            world_cfg,
            resources: list[str],
    ):
        StateHandler.__init__(self)
        BaseActor.__init__(self)
        Utils.__init__(self)

        self.available_actors = {
            "GEM": self.create_gem_worker,
            "FBRTDB": self.create_fbrtdb_worker,
            "SPANNER_WORKER": self.create_spanner_worker,
            "BQ_WORKER": self.create_bq_worker,
        }
        self.bq_tables=[]
        self.crcs_creator = CloudRcsCreator()

        self.resources = resources
        self.domain = "http://127.0.0.1:8001" if os.name == "nt" else f"https://{DOMAIN}"
        self.get_neighbors_endp = "/get-neighbors"
        self.sp_upsert_endp = "/sp/upsert"
        self.bq_upsert_endp = "/bq/upsert"
        self.sp_create_graph_endp = "/sp/create-graph"
        self.get_table_entry_endp = "/get-table-entry"
        self.alive_workers = []

        if "SPANNER_WORKER" in self.resources:
            self.sem = SpannerEmulatorManager()
            self.sem.main()
        self.world_cfg = world_cfg
        print("CLOUD MASTER initiaized")


    def get_bq_table_names_to_create(self):
        return ray.get(
            get_actor(name="UTILS_WORKER")
        ).get_nodes.remote(
            filter_key="type",
            filter_value=ALL_SUBS,
            just_id=True,
        )


    async def upsert_field_content(self):
        #  upsert vertical all ntypes to spanner AND
        # each nid to bq
        async def upsert_fields(ntype):
            try:
                await self.apost(
                    url=f"{self.domain}{self.sp_upsert_endp}",
                    data=dict(
                        table_name=ntype,
                        rows=[],
                    ),
                )

                # Upsert BQ (each nid gets one table
                await asyncio.gather(*[
                    self.apost(
                        url=f"{self.domain}{self.bq_upsert_endp}",
                        data=dict(
                            table=ntype,
                            rows=[row]
                        ),
                    )
                    for row in []
                ])
            except Exception as e:
                print(f"Err upsert_field_content: {e}")

        await asyncio.gather(*[
            upsert_fields(ntype)
            for ntype in ALL_SUBS
        ])


    def create_gem_worker(self, name):
        ref = Gem.options(
            lifetime="detached",
            name=name,
        ).remote()
        return ref


    async def create_actors(self):
        try:
            print("============= CREATE CLOD WORKERS =============")
            for actor_id in self.resources:
                self.create_worker(
                    name=actor_id,
                )

            print("cloud workers created")

            self.await_alive(
                id_map=self.resources
            )

            print("cloud workers alive")

            await self.create_rcs_wf()
            print("Exit CloudCreator...")
        except Exception as e:
            print(f"Error creating cloud workers: {e}")

    async def create_rcs_wf(self):
        """
        Create Cloud Acotrs, resources and fill them
        """
        print("========== CREATE CLOUD RCS ==========")
        try:
            if "SPANNER_WORKER" in self.resources:
                await self.crcs_creator.create_spanner_rcs()

            elif "BQ_WORKER" in self.resources:
                await self.crcs_creator.create_bq_rcs(
                    tables_to_create=[TABLE_NAME]
                )

            print("All Cloud Rcs created successfully")
        except Exception as e:
            print(f"Err create_rcs_wf: {e}")

    async def create_sgraph(self):
        await self.db_workers_ready()

        # Upsert Data
        await asyncio.gather(*[
            self.upsert_pixel(),
            self.upsert_edges(),
            self.upsert_init_fields(),
        ])

        # UPSERT ALL_SUBS FIELD PAYLOAD
        await self.upsert_field_content()

    async def upsert_pixel(self):
        try:
            # Upsert Pixel payload
            payload = {
                "admin_data": {
                    "rows": [
                        attrs
                        for _, attrs in self.world_creator.px_creator.g.G.nodes(data=True)
                        if attrs["type"] == self.world_creator.px_creator.layer
                    ],
                    "table_name": self.world_creator.px_creator.layer
                }
            }
            # upsert
            await self.apost(
                url=f"{self.domain}{self.sp_upsert_endp}",
                data=payload
            )
        except Exception as e:
            print(f"Err upsert_pixel: {e}")



    async def db_workers_ready(self):
        print("Await DB workers ready")
        # Upsert when resources are ready
        while self.bq_ref is None or self.sp_ref is None:
            try:
                self.bq_ref = serve.get_app_handle("BQ_WORKER")
                await self.bq_ref.ping.remote()

                self.sp_ref = serve.get_app_handle("SPANNER_WORKER")
                await self.sp_ref.ping.remote()

            except Exception as e:
                print(f"Err db_workers_ready: {e}")
            await asyncio.sleep(1)


    async def create_sgraph(self):
        # Upsert Pixel payload
        print("Create SGRAPH")
        payload = {
            "graph_name": ENV_ID,
            "node_tables": ALL_SUBS,
            "edge_tables": ["EDGES"],
        }

        # upsert
        await self.apost(
            url=f"{self.domain}{self.sp_create_graph_endp}",
            data=payload
        )
        print("SGRAPH created successfully")


    def create_worker(self, name):
        print(f"Create worker {name}")
        retry = 3
        for i in range(retry):
            try:
                # Remove __px_id form name (if)
                ref = self.available_actors[name](name)
                self.await_alive(["UTILS_WORKER"])

                if isinstance(ref, DeploymentHandle):
                    pass
                else:
                    ref = ref._ray_actor_id.binary().hex()

                ray.get_actor(name="UTILS_WORKER").set_node.remote(
                    dict(
                        nid=name,
                        ref=ref,
                        type="ACTOR",
                    )
                )
                break
            except Exception as e:
                print(f"Err: {e}")

    def create_spanner_worker(self, name):
        ref = SpannerWorker.options(
            name=name,
        ).remote()
        print("SPANNER worker deployed")
        return ref

    def create_bq_worker(self, name):
        ref = BQService.options(
            name=name,
            lifetime="detached"
        ).remote()

        print("BigQuery worker deployed")
        return ref

    def create_fbrtdb_worker(self, name):
        ref = FBRTDBAdminWorker.options(
            name="FBRTDB",
            lifetime="detached"
        ).remote()
        # Build G from admin_data
        return ref


    async def upsert_edges(self):
        # Upsert Pixel Edges to spanner
        try:
            payload = {
                "admin_data": {
                    "rows": [
                        {
                            "src": src,
                            "trgt": trgt,
                            **attrs,
                        }
                        for src, trgt, attrs in self.world_creator.px_creator.g.G.edges(data=True)
                    ],
                    "table_name": "EDGES"
                }
            }
            # upsert
            await self.apost(
                url=f"{self.domain}{self.sp_upsert_endp}",
                data=payload
            )
        except Exception as e:
            print(f"Err upsert_edges: {e}")



    async def upsert_init_fields(self):
        pixel_content = []
        try:
            for ntype in ALL_SUBS:
                for i in range(self.world_cfg["amount_nodes"]):
                    pixel_content.extend(
                        self.get_raw_attrs(
                            ntype,
                            px_index=i
                        )
                    )

                payload = {
                    "admin_data": {
                        "rows": pixel_content,
                        "table_name": ntype
                    }
                }

                # upsert
                await asyncio.gather(
                    *[
                        self.apost(
                            url=f"{self.domain}{self.bq_upsert_endp}",
                            data=payload
                        ),
                        self.apost(
                            url=f"{self.domain}{self.sp_upsert_endp}",
                            data=payload
                        )
                    ]
                )
        except Exception as e:
            print(f"Err upsert_init_fields: {e}")


if __name__ == "__main__":
    ref = CloudMaster.remote(
        resources=dict(
            SPANNER_WORKER=dict(),
            BQ_WORKER=dict()
        ),
        head=None
    )

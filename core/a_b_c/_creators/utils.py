import asyncio
import os
import ray
from ray import get_actor

from app_utils import ENV_ID, DOMAIN, DB_NAME
from qf_utils.all_subs import ALL_SUBS
from utils.utils import Utils

class CloudRcsCreator(Utils):

    def __init__(self):
        super().__init__()
        self.domain = "http://127.0.0.1:8001" if os.name == "nt" else f"https://{DOMAIN}"

    async def create_spanner_rcs(self):
        print("============== CREATE SPANNER RCS ===============")
        data = await self.get_sp_create_payload()
        for endpoint, payload in data.items():
            try:
                if isinstance(payload, list):
                    await asyncio.gather(*[
                        self.create_sp_rcs(endpoint, p)
                        for p in payload
                    ])

                elif isinstance(payload, dict):
                    await self.create_sp_rcs(endpoint, payload)
            except Exception as e:
                print(f"Err create_spanner_rcs: {e}")
        print("All Spanner rcs created successfully")


    async def create_sp_rcs(self, endpoint, payload):
        await self.apost(
            url=f"{self.domain}{endpoint}",
            data=payload)
        print(f"{endpoint} request finished")


    async def create_bq_rcs(
            self,
            tables_to_create:list[str]):
        print("create_bq_rcs")
        # todo start stop
        ref = get_actor(name="BQ_WORKER")
        ray.get(ref.create_database.remote(
            dict(
                db_name=DB_NAME
            )
        ))
        ref.create_table.remote(
            dict(
                table_names=[
                    *tables_to_create,
                ],
                ttype="node",
                attrs=None  # await self.node_attrs(),
            )
        )
        print("finsihed bq rcs creation")




    async def create_bq_rcs_web(self, tables_to_create):
        print("============== CREATE BQ RCS ===============")
        data = await self.get_bq_create_payload(tables_to_create)
        for endpoint, data in data.items():
            try:
                await self.apost(
                    url=f"{self.domain}{endpoint}",
                    data=data,
                )
                print(f"{endpoint} finalized")
            except Exception as e:
                print(f"Err create_bq_rcs: {e}")


    async def get_node_schema(self):
        # extract node attrs of ferm, gauge and h each
        node_attrs = ray.get(ray.get_actor(
            name="UTILS_WORKER"
        ).get_node_sum.remote())
        print("received node attrs:", node_attrs.keys())
        ##pprint.pp(node_attrs)

        schema = await self.apost(
            url=f"{self.domain}/sp/extract-schema",
            data={"attrs": node_attrs}
        )
        print("node schema received", schema)
        # unpack data
        return schema

    async def node_attrs(self):
        print("bq_schema")
        node_attrs = ray.get(ray.get_actor(
            name="UTILS_WORKER"
        ).get_node_sum.remote())
        print("received node attrs:", node_attrs.keys())
        return node_attrs

    async def get_edge_attrs(self):
        # obj_ref: ObjectRef to all edges
        print("get_edge_attrs")
        edge_attrs = ray.get(
            ray.get_actor(
                name="UTILS_WORKER"
            ).get_all_edge_attrs.remote())
        return edge_attrs



    async def get_sp_create_payload(self):
        node_table_schema = await self.node_attrs()
        edge_table_schema = await self.get_edge_attrs()
        data = {
            "/sp/create-instance": dict(
                instance_id=f"I_{ENV_ID}"
            ),
            "/sp/create-table": [
                dict(
                    table_name_map=[*ALL_SUBS, "PIXEL"],
                    attrs=node_table_schema,
                ),
                dict(
                    table_name_map=["EDGES"],
                    attrs=edge_table_schema,
                ),
            ],
            "/sp/create-change-stream": dict(
                table_names=ALL_SUBS,
                stream_type="node",
            ),
        }
        return data


    def get_edge_data(self):
        print("RELAY: Get edges")
        edge_attrs = ray.get(
            ray.get_actor(
                name="UTILS_WORKER"
            ).get_all_edge_attrs.remote()
        )
        return edge_attrs


    async def get_bq_create_payload(
            self, tables_to_create:list
    ):
        """
        Database
        Tables
        """
        web_data = {
            "/bq/create-database": dict(
                db_name=DB_NAME
            ),
            "/bq/create-table": dict(
                table_names=[
                    *tables_to_create,
                ],
                ttype="node",
                attrs=None#await self.node_attrs(),
            ),
        }



        return web_data


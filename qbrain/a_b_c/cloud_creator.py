import asyncio
import os
import time
import ray
from ray import serve

from qbrain.a_b_c.bq_agent.bq_worker import BQService
from qbrain.a_b_c.spanner_agent.spanner_agent import SpannerWorker
from qbrain.core.app_utils import ENV_ID, DOMAIN
from qbrain.a_b_c.db_worker import FBRTDBAdminWorker
from qbrain.qf_utils.all_subs import ALL_SUBS
from qbrain.utils.utils import Utils


@ray.remote
class CloudMaster(Utils):

    def __init__(self, head, resources: list[str]):
        super().__init__()
        self.head = head

        self.host = {}

        self.rc_base_id = ENV_ID.upper()

        self.resources = resources
        self.alive_workers = []

        self.domain = "http://127.0.0.1:8001" if os.name == "nt" else f"https://{DOMAIN}"

        self.available_actors = {
            "FBRTDB_ADMIN_WORKER": self.create_fbrtdb_worker,
            "SPANNER_WORKER": self.create_spanner_worker,
            "BQ_WORKER": self.create_bq_worker,
        }
        self.create_actors()

    def create_actors(self):
        for actor_id in self.resources:
            self.create_worker(
                name=actor_id,
            )

        self.await_alive()

        print("Cloud Db Creator Creation finished")

        self.head.handle_initialized.remote(self.host)

        # Create cloud resources
        asyncio.run(
            self.create_rcs_wf(

            )
        )
        print("Exit CloudCreator...")
        ray.actor.exit_actor()


    def get_bq_create_payload(self):
        """
        Database
        Tables
        """
        data = {
            "/create-database": dict(
                db_name=ENV_ID
            ),
            "/create-table": [ALL_SUBS, "edges"]
        }
        return data


    async def get_node_schema(self):
        obj_ref = ray.get(self.host["UTILS_WORKER"].get_nodes_each_type.remote())
        schema_ref = await self.apost(
            url=f"{self.domain}/extract-schema/",
            data={"type": "node", "obj_ref": obj_ref}
        )
        # unpack admin_data
        schema = ray.get(schema_ref)
        return schema

    async def get_edge_schema(self, obj_ref):
        # obj_ref: ObjectRef to all edges
        schema_ref = await self.apost(
            url=f"{self.domain}/extract-schema/",
            data={
                "type": "edge",
                "obj_ref": obj_ref
            }
        )
        #unpack admin_data
        schema = ray.get(schema_ref)
        return schema

    async def get_sp_create_payload(self):
        edge_obj_ref, eids = self.get_edge_data()

        # Fetch schema from admin_data[list]
        schemas = await asyncio.gather(
            *[
                self.get_edge_schema(obj_ref=edge_obj_ref),
                self.get_node_schema(),
            ]
        )

        G_ref = ray.get(self.host["UTILS_WORKER"].get_G.remote())

        data = {
            "/create-instance": dict(
                instance_id=f"I_{self.rc_base_id}"
            ),
            "/create-rcs": dict(
                instance_id=f"I_{self.rc_base_id}",
                node_table_map=[*ALL_SUBS, "PIXEL"],
                edge_table_map=eids,
                edge_table_schema=schemas[0],
                node_table_schema=schemas[1],
                graph_name=f"G_{self.rc_base_id}",
            ),
            "/load-init-state-db-from-nx": dict(
                nx_obj_ref=G_ref
            ),
            "/create-change-stream": dict(
                node_tables=ALL_SUBS,
                edge_tables=eids,
            ),
        }
        return data

    async def get_edge_data(self):
        print("RELAY: Get edges")
        edge_refs = ray.get(self.host["UTILS_WORKER"].get_all_edges.remote(
            datastore=False,
            just_id=False,
        ))

        edges: list[dict] = ray.get(edge_refs)
        eids = [eid.get("id").upper() for eid in edges]
        new_obj_ref = ray.put(edges)
        return new_obj_ref, eids


    async def create_spanner_rcs(self):
        print("============== CREATE SPANNER RCS ===============")
        data = await self.get_sp_create_payload()
        for endpoint, data in data.items():
            response = await self.apost(
                url=f"{self.domain}{endpoint}",
                data=data,
            )
            print("response.admin_data", response.data)
            if response.ok:
                continue
            else:
                # todo error intervention
                continue

    async def create_bq_rcs(self):
        print("============== CREATE BQ RCS ===============")
        data = self.get_bq_create_payload()
        for endpoint, data in data.items():
            response = await self.apost(
                url=f"{self.domain}{endpoint}",
                data=data,
            )
            print("response.admin_data", response.data)


    async def create_rcs_wf(self):
        await self.create_spanner_rcs()
        await self.create_bq_rcs()


    def create_worker(self, name):
        print(f"Create worker {name}")
        retry = 3
        for i in range(retry):
            try:
                # Remove __px_id form name (if)
                ref = self.available_actors[name](name)
                return ref
            except Exception as e:
                print(f"Err: {e}")

    def create_spanner_worker(self, name):
        ref = serve.run(SpannerWorker.options(
            name=name,
            lifetime="detached"
        ).bind(),
            name=name,
            route_prefix=f"/spanner"
            )
        print("SPANNER worker deployed")
        self.host[name] = ref

    def create_bq_worker(self, name):
        ref = serve.run(
            BQService.options(
                name=name,
                lifetime="detached"
            ).bind(),
                name=name,
        )
        print("BigQUERY worker deployed")
        self.host[name] = ref

    def create_fbrtdb_worker(self, name):
        ref = FBRTDBAdminWorker.options(
            name=name,
            lifetime="detached"
        ).remote()
        self.host[name] = ref
        # Build G from admin_data
        return ref

    def await_alive(self):
        while len(self.alive_workers) != len(list(self.resources)):
            for nid, ref in self.host.items():
                try:
                    if nid not in self.alive_workers:
                        # Ping the actor with a trivial call
                        alive: bool = ray.get(
                            ref.ping.remote(),
                            timeout=5
                        )
                        print(f"Actor {nid} is alive")
                        if alive is True:
                            self.alive_workers.append(nid)
                        time.sleep(2)
                except Exception as e:
                    print(f"Actor is dead or unreachable: {e}")
        print(f"{len(self.alive_workers)} / {len(list(self.resources))} ALIVE ")
        return True


if __name__ == "__main__":
    ref = CloudMaster.remote(
        resources=dict(
            SPANNER_WORKER=dict(),
            BQ_WORKER=dict()
        ),
        head=None
    )

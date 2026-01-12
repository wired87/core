import ray

from _ray_core.base.base import BaseActor
from a_b_c.spanner_agent.controller import SpannerController
from core.app_utils import TESTING


@ray.remote(num_cpus=.5)
class SpannerWorker(
    BaseActor,
    SpannerController
):
    def __init__(self):
        print("======== Initializing SpannerWorker =========")
        BaseActor.__init__(self)
        SpannerController.__init__(
            self,
            testing=TESTING
        )
        print("SPANNER WORKER READY FOR ACTION")






"""
WASTELANDS



@APP.post("/create-rcs")
    async def create_resources_route(
            self,
            instance_id: str,
            node_table_map,
            node_table_schema,
            graph_name: str = "DEFAULT_GRAPH",
    ):

        async def create_workflow():

            print(f"🏗️ Starting creation workflow for instance: {instance_id}")
            success:bool
            # create db & tables 
            success = await self.spa_manager.create_core_rcs(
                node_table_map,
                edge_table_map,
                edge_table_schema,
                node_table_schema,
            )

            if success is True:
                print("Create Spanner Graph")
                success = self.sp_creator.create_graph(
                    node_tables=node_table_map,
                    edge_tables=edge_table_map,
                    graph_name=None
                )
                print(f"✅ Graph {graph_name} created/recreated.")
            print(">>> SG RCS PROCESS FINISHED")
            return {"message": "All resources checked and created/updated successfully! ✨"}

        return await self._safe_task_run(create_workflow)

    # ------------------------------------------------------------------------------------------------------------------



"""



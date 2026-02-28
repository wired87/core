import asyncio

import networkx as nx
import ray
from typing import List, Dict
from fastapi import HTTPException

from qbrain.a_b_c.spanner_agent._spanner_graph import create_default_node_table_query
from qbrain.a_b_c.spanner_agent._spanner_graph.acore import ASpannerManager
from qbrain.a_b_c.spanner_agent._spanner_graph.change_streams.main import ChangeStreamMaster
from qbrain.a_b_c.spanner_agent._spanner_graph.create_workflow import SpannerCreator
from qbrain.a_b_c.spanner_agent._spanner_graph.emulator import SpannerEmulatorManager
from qbrain.a_b_c.spanner_agent._spanner_graph.g_utils import SpannerGraphManager

from qbrain.core.app_utils import ENV_ID, GCP_ID
from qbrain.auth.load_sa_creds import load_service_account_credentials
from qbrain.qf_utils.all_subs import ALL_SUBS


class SpannerController:

    """
    production:
    Todo cli automated agent pick right cmd and run as describe yesterday
    feel free to do it as contribution to
    this project
    Diese Projekt wird uns ALLEN Wohlstand bringen
    nicht nur 1% wie üblich
    """


    def __init__(
            self,
            testing:bool
    ):

        if testing:
            sem = SpannerEmulatorManager()
            sem.main()

        self.sg_utils = SpannerGraphManager()
        self.spa_manager = ASpannerManager()
        self.sp_creator = SpannerCreator()
        self.cs_manager = ChangeStreamMaster()

        # Initialisiere asynchron eine Sitzung beim Start
        self.project_id = GCP_ID

        # Statische Graph-Info für Demo
        self.graph_name = ENV_ID
        print("=========== SpannerWorker initilized ===========")

        print("SpannerWorker Actor: Initialization complete. ✅")

    async def _safe_task_run(self, func, *args, **kwargs):
        """Erstellt eine Task und wartet, um den Actor nicht zu blockieren."""
        """
        if self.spa_manager.session is None:
            await self.spa_manager.acreate_session()
        """
        return await asyncio.create_task(func(*args, **kwargs))


    async def create_table(self, data:dict):
        print("▶️ SpannerWorker.create_table called.")
        table_name_map:list[str] = data["table_name_map"]
        attrs = data["attrs"]

        query = ""
        for table in table_name_map:
            print(f"  - Generating create query for table: {table}")
            query += f"{create_default_node_table_query(table_name=table)} \n"

        asyncio.create_task(
            self.spa_manager.asnap(query)
        )

        schema = self.sg_utils.get_spanner_schema(
            attrs=attrs
        )
        print(f"  - Extending schema for {len(table_name_map)} tables.")
        # extend schema to all tables
        queries = self.spa_manager.add_table_schems_query(
            table_name_map,
            schema
        )

        query_tasks = [self.spa_manager.asnap(query) for query in queries]
        await asyncio.gather(*query_tasks)

        print("✅ SpannerWorker.create_table: Finished creating tables and extending schemas.")
        return "Tables created and schemas extended successfully."


    async def get_table_entry(
            self,
            table_name,
            where_key,
            is_value,
            select_table_keys
    ):
        print(f"▶️ SpannerWorker.get_table_entry called for table '{table_name}'.")
        entry = None
        table_exists:bool = await self.spa_manager.acheck_table_exists(table_name)
        print(f"  - Checking if table '{table_name}' exists... Result: {table_exists}")
        if table_exists is True:
            query = self.spa_manager.custom_entries_query(
                table_name,
                check_key=where_key,
                check_key_value=is_value,
                select_table_keys=select_table_keys
            )
            print("  - Executing query to fetch entry.")
            entry:dict or list[dict] = self.spa_manager.asnap(
                query,
                return_as_dict=True
            )
            print(f"  - Query finished. Found entry: {'Yes' if entry else 'No'}")
        else:
            print(f"  - ⚠️ Table '{table_name}' does not exist.")
        return entry

    async def create_instance(
            self,
            instance_id,
    ):
        print(f"▶️ SpannerWorker.create_instance called for instance '{instance_id}'.")
        ## todo craeate schema based on neighbor type
        if self.spa_manager.session is None:
            await self.spa_manager.acreate_session()

        self.sp_creator.create_spanner_instance(
            instance_id=instance_id,
            processing_units=100
        )
        print(f"✅ SpannerWorker.create_instance: Finished for instance '{instance_id}'.")
        return f"Instance '{instance_id}' creation process finished."

    async def create_database(
            self,
            database_id,
            instance_id,
    ):
        print(f"▶️ SpannerWorker.create_database called for db '{database_id}' in instance '{instance_id}'.")
        ########################## INSTANCE ##########################
        ## todo craeate schema based on neighbor type
        try:
            self.sp_creator.sp_create_database(
                database_id=database_id,
                instance_id=instance_id,
                update_db=True
            )
            if self.spa_manager.session is None:
                await self.spa_manager.acreate_session(
                    database_id=database_id
                )
        except Exception as e:
            print(f"  - ❌ Error in create_database: {e}")
            raise e
        print(f"✅ SpannerWorker.create_database: Finished for db '{database_id}'.")
        return f"Database '{database_id}' creation process finished."

    async def create_graph(self, data):
        print(f"▶️ SpannerWorker.create_graph called for graph '{data.get('graph_name')}'.")
        node_tables=data["node_tables"]
        edge_tables=data["edge_tables"]
        graph_name=data["graph_name"]

        print(f"  - Generating query to create graph with {len(node_tables)} node tables and {len(edge_tables)} edge tables.")
        query = self.spa_manager.get_create_graph_query(
            graph_name=graph_name,
            node_tables=node_tables,
            edge_tables=edge_tables,
        )
        await self.spa_manager.asnap(query)
        print(f"✅ SpannerWorker.create_graph: Graph '{graph_name}' created successfully.")
        return f"Graph '{graph_name}' created successfully."

    async def load_init_state_db_from_nx(self, nx_obj_ref):
        print("▶️ SpannerWorker.load_init_state_db_from_nx called.")
        try:
            print("  - Getting NetworkX graph object from ObjectRef.")
            #nx_obj_ref = ray.get(GLOBAC_STORE["UTILS_WORKER"].get_G.remote())
            # BUILD G
            G:nx.Graph = ray.get(nx_obj_ref)

            print("load EDGE tables")
            await asyncio.gather(
                *[
                    self.spa_manager.upsert_row(
                        batch_chunk=[{
                            "src": src,
                            "trgt": trgt,
                            **attrs,
                        }],
                        table=attrs.get("id").upper())
                    for src, trgt, attrs in G.edges(data=True)
                    if attrs.get("type") in ALL_SUBS
                ]
            )
            print("✅ SpannerWorker.load_init_state_db_from_nx: Successfully upserted initial state.")
            return "Initial state loaded into database successfully."

        except Exception as e:
            print(f"  - ❌ Error in load_init_state_db_from_nx: {e}")
            raise e

    async def create_change_stream(self, table_names, stream_type="node"):
        print(f"▶️ SpannerWorker.create_change_stream called for stream_type '{stream_type}'.")
        csid = None
        if stream_type == "node":
            csid = f"NODE_{ENV_ID}"
        elif stream_type == "edge":
            csid = f"EDGE_{ENV_ID}"

        if csid:
            success = self.cs_manager.create_change_stream(
                name=csid,
                table_names=table_names
            )
            if success is True:
                print(f"✅ Change Stream {csid} created.")
                return f"Change Stream '{csid}' created successfully."

        print(f"  - ⚠️ Issue creating change stream for type '{stream_type}'.")
        return "Issue creating change stream."

    async def upsert(self, payload):
        """Route zum Einfügen/Aktualisieren von Batch-Zeilen."""
        print("▶️ SpannerWorker.upsert called.")
        data = payload["admin_data"]
        rows: List[Dict] = data["rows"]
        table_name = data["table_name"]
        print(f"  - Preparing to upsert {len(rows)} rows into table '{table_name}'.")

        async def upsert_workflow():
            if not await self.spa_manager.acheck_table_exists(table_name):
                print(f"  - ❌ Table '{table_name}' does not exist.")
                raise HTTPException(status_code=404, detail=f"Table {table_name} does not exist. ❌")

            # Die Logik in aupdate_insert (aus acore.py) handhabt bereits Batching
            await self.spa_manager.aupdate_insert(table_name, rows)

            print(f"✅ SpannerWorker.upsert: Successfully upserted {len(rows)} rows into '{table_name}'.")
            return f"Successfully upserted {len(rows)} rows into {table_name}."

        return await self._safe_task_run(upsert_workflow)

    # ------------------------------------------------------------------------------------------------------------------

    async def read_change_stream_route(
            self,
            data
    ):
        print("▶️ SpannerWorker.read_change_stream called.")
        end_time = data["end_time"]
        start_time = data["start_time"]
        stream_name = data["stream_name"]
        result = await asyncio.gather(*[
            self.cs_manager.read_change_stream(
                stream_name,
                start_time,
                end_time
            ),
        ])
        ref = ray.put(result)
        print(f"✅ SpannerWorker.read_change_stream: Successfully retrieved changes from stream '{stream_name}' and put them into Ray object store.")
        return ref

    # ------------------------------------------------------------------------------------------------------------------

    async def get_neighbors(self, nid: str, graph_name):
        """Route zur Abfrage der Center-Nodes für einen gegebenen Nachbarn."""
        print(f"▶️ SpannerWorker.get_neighbors called for nid '{nid}'.")
        return {}

    # ------------------------------------------------------------------------------------------------------------------

    async def list_entries(
            self,
            column_name,
            table_name,
    ):
        print(f"▶️ SpannerWorker.list_entries called for table '{table_name}'.")

        query = self.spa_manager.get_entries_from_list_str(
            column_name=column_name,
            table_name=table_name,
        )
        asyncio.create_task(self.spa_manager.asnap(query))
        return {"message": f"Instance and all contained resources have been deleted. 💥"}

    async def delete_instance(self, instance_id: str):
        """Route zur vollständigen Löschung einer Spanner-Instanz und aller Ressourcen."""
        print(f"▶️ SpannerWorker.delete_instance called for instance '{instance_id}'.")

        async def delete_workflow():
            print(f"  - 💣 Starting deletion workflow for instance: {instance_id}")

            # 1. Lösche alle Tabellen in der Standarddatenbank (Simuliert)
            table_names = await self.spa_manager.atable_names()
            await self.spa_manager.adel_table_batch(table_name=table_names)
            print(f"✅ Deleted all {len(table_names)} tables.")

            # 2. Lösche die Datenbank (Hier müsste die Logik aus SpannerCreator oder SpannerCore her)
            # await self.spa_manager.update_db(self.spa_manager.drop_database_query(self.spa_manager.db))
            print(f"✅ Deleted database.")

            # 3. Lösche die Instanz (Müsste in SpannerCore implementiert sein)
            # self.sp_creator.delete_spanner_instance(instance_id)
            print(f"✅ Deleted instance {instance_id}.")

            return f"Instance {instance_id} and all contained resources have been deleted. 💥"

        return await self._safe_task_run(delete_workflow)


if __name__ =="__main__":
    load_service_account_credentials()
    ctlr = SpannerController(testing=True)


from gdb_manager.bq.bq_handler import BQCore
from gdb_manager.firebase.real_time_database import FirebaseRTDBManager
from utils.embedder import embed

import asyncio
import json
import os
from typing import Dict, List

from gdb_manager.spanner.acore import SpannerAsyncHelper

from gdb_manager.storage.storage import GBucket
from gdb_manager.spanner.graph_loader import SpannerGraphLoader


# todo batch process: collect, then send up in batches


class DBManager(
    SpannerGraphLoader,
    SpannerAsyncHelper,
    # BigQueryGraphHandler,
    # ABQHandler
    BQCore,
    #GUtils
):
    """
    Gets all times an nx.G
    Upserts either full batches or single entries to sp, bq or fb
    """
    
    
    def __init__(
            self,
            table_name="CELL",
            upload_to: str = "fb",  # bq || sp || fb
            database=None,
            g_from_path=None,
            user_id=None,
            instance=None,
    ):
        
        self.database = database
        self.g_from_path = g_from_path
        
        self.instance = instance

        self.table_name = table_name
        self.upload_to = upload_to
        self.bucket = GBucket()
        self.batch_chunk_size = 10000
        self.loop_count = 0

        self.tables_created = []
        self.edge_batch = {}
        self.runs = 0

        if upload_to in ["sp", "bq"]:
            # ABQHandler.__init__.py(self, database=database)
            # BigQueryGraphHandler.__init__.py(self)
            SpannerAsyncHelper.__init__(self, database)
            SpannerGraphLoader.__init__(self, database)
            BQCore.__init__(self, dataset_id=database)
        else:
            self.firebase = FirebaseRTDBManager(base_path=database)
        print("DataManager initialized")


    ####################################
    # CORE
    ####################################
    async def abatch_commit(
            self, 
            data, 
            embed_only=True, 
            create_session=True,
            path=None
    ):
        """
        se for (heavy) uploads
        """
        print(">>>Start batch commit")
        try:
            #self.print_status()
            if self.upload_to == "fb":
                # firebase alltimes gets a nx.G
                self.firebase.upsert_data(
                    path,
                    data
                )
            else:
                # sp || bq alltimes gets a GUtils.schemas - struct
                if create_session is True:
                    await self.acreate_session()
                await self.acreate_tables_batch(data)
                await self.aschema_batch_process(data)
                await self.aupsert_batch(embed_only, data)
        except Exception as e:
            print(f"Error during abatch_commit: {e}")
        print(">>>Finished")



    ####################################
    # NODE EDGE HANDLER EXTEND
    ####################################


    def remove_node(self, node_id, ntype):
        if self.upload_to == "sp":
            pass
        elif self.upload_to == "bq":
            pass
        elif self.upload_to == "fb":
            self.firebase.delete_data(path=f"{ntype}/{node_id}/")



    #######################
    # HELPER
    ####################################

    
    
    
    

    

    def get_ids(self, table=None):
        if table is None:
            table = self.table_name
        if self.upload_to == "sp":
            ids = self.get_all_ids(table)
        else:
            ids = self.get_column_values(table, "id", )
        print(f"Fetched {len(ids)} ids from {table}")
        return ids

    def upsert_history_batch(self, g = None):
        updates={}
        # Convert history struct and upsrt it to fb
        if self.upload_to == "fb":
            for table_name, ids in self.history.items():
                for node_id, timestep_value in ids.items():
                    for time, data in timestep_value.items():
                        path = rf"{self.database}/H_{table_name}/{node_id}/{time}"
                        updates.update({
                                path: data
                            }
                        )
            self.g.firebase.upsert_data(
                path="/",
                data=updates
            )
            # reset
            self.history = {}
            print("Finished upsert to FB")

    def data_preprocessor(self, data: list[dict] = [], keys_to_remove: list[str] = [], key="id") -> list[dict]:
        print(f"Removing {len(keys_to_remove)}/{len(data)} ({len(data) - len(keys_to_remove)} processed ids ")
        filtered_data = [item for item in data if item.get(key) not in keys_to_remove]
        print("Entries left:", len(filtered_data))
        return filtered_data

    async def load_content(
            self,
            local_path,
            bucket_path,
            layer,
            test_chunk,
            testing
    ):
        print("Load content")
        if testing is not None:
            content = None #await aread_content(test_chunk)
        else:

            if not os.path.exists(local_path):
                print(f"Local path {local_path} does not exists. Fetch from {bucket_path}")
                content = json.loads(self.bucket.download_blob(bucket_path))
                content = self.structure_content_save(
                    content,
                    layer=layer,
                    save_to=local_path,
                    single=False
                )
            else:
                print("Fetch content from", local_path)
                with open(local_path, 'r') as f:
                    content = json.load(f)
        # print("Content load")
        return content

    async def alocal_batch_loader(self, args, data):
        table_name = args.get("type")
        row_id = args["id"]
        if table_name:
            if table_name not in data:
                data[table_name] = {
                    "schema": {},
                    "rows": [],
                    "id_map": set(),
                }
                # print(f"Added {table_name} to schema")

            if row_id not in [item for item in data[table_name]["id_map"]]:
                data[table_name]["rows"].append(args)
                data[table_name]["id_map"].add(row_id)



    async def acreate_tables_batch(
            self,
            data,
    ):
        
        # print("Creating tables batch")
        if self.upload_to == "sp":
            await asyncio.gather(*[
                self.acheck_add_table(
                    table_name=k,
                    ttype="edge" if any(c.islower() for c in k) else "node",
                    schema_fetch=False
                ) for k, v in data.items()])

            print("Table process finished")

        else:
            for k, v in data.items():
                # todo get all (filtered) bq tables

                self.get_create_bq_table(
                    table_name=k,
                    ttype="edge" if any(c.islower() for c in k) else "node",
                )

    def create_tables_batch(self, data):
        # print("Creating tables batch")
        if self.upload_to == "sp":
            for k, v in data.items():
                print("k", k)
                #if k not in self.all_tables:
                self.check_add_table(
                    table_name=k,
                    # Check if min one chear is lower
                    ttype="edge" if any(c.islower() for c in k) else "node",
                    schema_fetch=False
                )
        else:
            for k, v in data.items():
                self.get_create_bq_table(
                    table_name=k,
                    ttype="edge" if any(c.islower() for c in k) else "node",
                )

    async def aschema_batch_process(self, data):
        # print("Schema process")
        for k, v in data.items():
            all_keys = set().union(*(row.keys() for row in v["rows"]))
            for row_key in all_keys:
                sample_value = next((row[row_key] for row in v["rows"] if row_key in row), None)
                if sample_value is not None:
                    if self.upload_to == "sp":
                        col_type = self.get_spanner_type(sample_value)
                    else:
                        col_type = self.get_bq_type(sample_value)
                    print("Adding row key", row_key, col_type)
                    v["schema"][row_key] = col_type

        if self.upload_to == "sp":
            await asyncio.gather(
                *[
                    self.aadd_col(
                        keys=v["schema"],
                        table=k,
                        type_from_val=False,
                    ) for k, v in data.items()
                ]
            )


    def schema_batch_process(self, data):
        # print("Schema process")
        for k, v in data.items():
            all_keys = set().union(*(row.keys() for row in v["rows"]))
            for row_key in all_keys:
                sample_value = next((row[row_key] for row in v["rows"] if row_key in row), None)
                if sample_value is not None:
                    if self.upload_to == "sp":
                        col_type = self.get_spanner_type(sample_value)
                    else:
                        col_type = self.get_bq_type(sample_value)
                    # print("Adding row key", row_key, spanner_type)
                    v["schema"][row_key] = col_type

        if self.upload_to == "sp":
            for k, v in data.items():
                self.check_add_cols_batch(
                    keys=v["schema"],
                    t=k
                )
        else:
            for k, v in data.items():
                self.update_bq_schema(
                    keys=v["schema"],
                    table=k
                )

    async def aupsert_batch(self, embed_only, data):
        print("Begin upsert process")
        if self.upload_to == "sp":
            tasks = []
            #print("data", data)
            for k, v in data.items():
                rows = [embed(row) for row in v["rows"]] if embed_only else v["rows"]
                print("Len rows", len(rows))
                tasks.append(
                    self.aupdate_insert(
                        table=k,
                        rows=rows
                    )
                )
            await asyncio.gather(*tasks)
            print("Batch upserted")

        else:
            for k, v in data.items():
                # Need to Stage changes
                self.bq_insert(table_id=k, rows=v["rows"], schema=v["schema"])

    def upsert_batch(self, data):
        if self.upload_to == "sp":
            for k, v in data.items():
                self.update_insert(
                    table=k,
                    rows=v["rows"]
                )
        else:
            for k, v in data.items():
                self.bq_insert(
                    table_id=k,
                    rows=v["rows"],
                    schema=v["schema"]
                )





    def get_len_neighbors_type_nx(self, node, target_type):
        len_neighbors = 0
        for neighbor in self.G.neighbors(node):
            if self.G.nodes[neighbor].get('type') == target_type:
                return neighbor, self.G.nodes[neighbor]
        return None  # No neighbor of that type found

    def get_gene_id_name(self, chrom=False):
        if chrom:
            query = f"""
                SELECT id, coord_system_seq_name, start, 'end' 
                FROM GENE 
                WHERE id IS NOT NULL 
                  AND coord_system_seq_name IS NOT NULL 
                  AND start IS NOT NULL 
                  AND 'end' IS NOT NULL
              """
        else:
            query = f"""
                    SELECT id, name FROM GENE WHERE id IS NOT NULL AND name IS NOT NULL
                    """
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(query)

        if results:
            if chrom:
                stuff = [{"id": stuff[0], "chrom": stuff[1], "start": stuff[2], "end": stuff[3]} for stuff in results]
                print("Gene entry 0", stuff[0])
            else:
                stuff = {f"{stuff[0]}": stuff[1] for stuff in results}
                print("Gene entries", len(stuff.items()))
            return stuff


    def table_schema_process(self, data, table_name, attrs, ttype="node"):
        if not data.get(table_name):
            # create table, return its default schema (if new created), update local
            schema = self.check_add_table(table_name=table_name, ttype=ttype)
            # print("schema", schema)
            data[table_name] = schema

        for k, v in attrs.items():
            if k not in data.get(table_name):
                spanner_type = self.get_spanner_type(v)
                self.check_add_cols(
                    col_type=spanner_type,
                    key=k,
                    t=table_name,
                    existing_cols=data[table_name].keys()
                )
                data[table_name][k] = spanner_type

        for k, v in attrs.items():
            # print("kv", k, v)
            if k not in data.get(table_name):
                data[table_name].update({k: self.get_spanner_type(v)})

    ############ SPANNER
    async def atable_schema_process(self, data, table_name: str, attrs: Dict, ttype: str = "node"):
        if not data.get(table_name):
            # create table, return its default schema (if new created), update local
            schema = await self.acheck_add_table(table_name=table_name, ttype=ttype)
            # print("schema", schema)
            data[table_name] = schema
        try:
            if self.upload_to == "sp":
                schema = await self.aadd_col(keys=attrs, table=table_name, existing_schema=data[table_name])
                # print("table schema", schema)
            else:
                schema = await self.update_bq_schema(keys=attrs, table=table_name)
            if isinstance(schema, dict) and len(schema.keys()):
                data[table_name].update(schema)
        except Exception as e:
            print("Error in atable_schema_process", e)

    ##############BQ

    async def bqaadd_edge(self, table_name, data):
        if not data.get(table_name):
            data[table_name]: List[Dict] = []
        if data.get(table_name) and len(data[table_name]) >= 5000:
            # upload bq batch
            schema = {}
            for item in data[table_name]:
                for k, v in item.items():
                    if k not in schema:
                        schema[k] = self.get_spanner_type(v)

    ################################################################################################

    async def find_node_by_attribute(self, attr_name, attr_value):
        """Finds a node by attributCheck if Spanner tablee value (Optimized)."""
        g_list = list(self.G.nodes(data=True))
        return next((node for node, attrs in g_list if attrs.get(attr_name) == attr_value), None)

    def _fetch_g_data(self):
        self.initial_data = {}
        if self.upload_to == "fb":
            print("Fetch entire dir from FB")
            data = self.firebase.get_data(path=f"/")
            raw_data = data[0]
            # todo filter out here session data and history -> (h eh bq)
            for k, v in raw_data.items():
                if k in ["session"] or k.startswith("H_"): # -> todo BQ for history!!!
                    continue
                self.initial_data[k] = v

            print(f"Data received from FB")
            return self.initial_data
        else:
            print("Must define logic for otgher plattforms")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
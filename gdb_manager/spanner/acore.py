import asyncio
import pprint
from typing import List, Dict

from google.cloud.spanner_admin_database_v1 import DatabaseAdminAsyncClient, UpdateDatabaseDdlRequest
from google.cloud.spanner_dbapi.parsed_statement import Statement
from google.cloud.spanner_v1 import SpannerAsyncClient, ExecuteBatchDmlRequest
from google.cloud.spanner_v1.services import spanner
from google.cloud.spanner_v1.types import (
    ExecuteSqlRequest,
    Mutation,
    CommitRequest,
    CreateSessionRequest,
)
from google.cloud.spanner_v1.types.transaction import TransactionOptions

from _google import GCP_ID
from _google.spanner import create_default_auth_table_query, create_default_edge_table_query, \
    create_default_node_table_query, SP_DATABASE_ID

from _google.spanner.graph_loader import SpannerCore


class SpannerAsyncHelper(SpannerCore):
    """
    If you need to change the structure of your database
    (add a column, create a table, etc.), use the DatabaseAdminAsyncClient.

    If you need to read or write data within your existing tables,
    use the Client or AsyncClient from the google.cloud.spanner library.
    """

    def __init__(self, db=None):
        super().__init__(db=db)
        self.db = db or SP_DATABASE_ID
        self.db_path = f"projects/{GCP_ID}/instances/brainmaster/databases/{self.db}"
        self.session = None

    async def acreate_session(self):
        """Creates and returns a Spanner session asynchronously."""
        self.adb_client = DatabaseAdminAsyncClient()
        self.aclient = SpannerAsyncClient()

        try:
            request = CreateSessionRequest(database=self.db_path)
            self.session = await self.aclient.create_session(request=request)
            print("Session created:", self.session.name)
            return self.session
        except Exception as e:
            print(f"Error creating session: {e}")
            return None

    async def acheck_table_exists(self, table_name: str, limit=0) -> bool:
        try:
            query = f"SELECT table_name FROM information_schema.tables WHERE table_name = '{table_name}'"
            request = ExecuteSqlRequest(session=self.session.name, sql=query)
            result = await self.aclient.execute_sql(request=request)
            return bool(result.rows)
        except Exception as e:
            print("Error occurred while executing:", e)
            if limit == 0:
                await self.acreate_session()
                limit = 1
                return await self.acheck_table_exists(table_name, limit=limit)

    async def upsert_row(self, table: str, batch_chunk: list[dict]):
        print("Batch chunk")
        pprint.pp(batch_chunk)
        print("len", batch_chunk)
        try:
            mutations = [
                Mutation(
                    insert_or_update=Mutation.Write(
                        table=table,
                        columns=list(row.keys()),
                        values=[list(row.values())]
                    )
                ) for row in batch_chunk
            ]

            await self.aclient.commit(
                request=CommitRequest(
                    session=self.session.name,
                    single_use_transaction=TransactionOptions(read_write={}),
                    mutations=mutations,
                )
            )
            # print(f"✅ Upserted row into {table}")
        except Exception as e:
            print("Error upsert row:", batch_chunk, e)

    async def aupdate_insert(self, table: str, rows: List[Dict]):
        if len(rows)>0:
            #print(f"Upsert {len(rows)} to {table}")
            for i in range(0, len(rows), self.batch_size):
                batch_chunk = rows[i:i + self.batch_size]
                try:
                    print("len batch_chunk", len(batch_chunk))
                    await self.upsert_row(table, batch_chunk)
                except Exception as e:
                    #await asyncio.sleep(3)
                    print(f"Error while UPDATING batch: {e}", )
                    await self.upsert_row(table, batch_chunk)

    async def asnap(self, query, return_as_dict=False):
        print("Query", query)
        #query="""SELECT * FROM GO WHERE meta_definition_val IS NOT NULL LIMIT 10"""
        request = ExecuteSqlRequest(session=self.session.name, sql=query)
        result = await self.aclient.execute_sql(request=request, timeout=999.0)

        if return_as_dict is True:
            #print("result rows", result.rows)
            schema_keys = [field.name for field in result.metadata.row_type.fields]
            rows_as_dict=[]
            #print("schema_keys", schema_keys)
            for row in result.rows:
                rows_as_dict.append({k: v for k,v in zip(schema_keys, row)})
            #pprint.pp(rows_as_dict)
            return rows_as_dict
        return result

    async def atable_names(self):
        """Get all table names"""
        query = self.table_names_query()
        result = await self.asnap(query)
        print("result", result.rows)
        return [row[0] for row in result.rows if not row[0] in self.info_schema]

    ##################### DB MANIPUTATION
    async def adel_table_batch(self, startswith=None, endswith=None, table_name=None):
        if table_name:
            query = self.drop_table_query(table_name)
            await self.update_db(query)
        else:
            table_names = await self.atable_names()
            del_tables = [
                table for table in table_names
                if table.startswith(startswith) and table.endswith(endswith)
            ]

            for i in range(0, len(del_tables), self.del_table_batch_size):
                chunk = del_tables[i:i + self.del_table_batch_size]
                print(f"🗑️ Deleting {len(chunk)} tables...")

                await asyncio.gather(*[
                    self.update_db(self.drop_table_query(table)) for table in chunk
                ])

                if i + self.del_table_batch_size < len(del_tables):
                    print(f"✅ Deleted {len(chunk)} tables. Sleeping 60 seconds to avoid quota limits...")
                    await asyncio.sleep(60)

    async def afetch_table_schema(self, table_name: str):
        try:
            query = self.table_schema_query(table_name)
            request = ExecuteSqlRequest(session=self.session.name, sql=query)
            result = await self.aclient.execute_sql(request=request)
            schema = {row[0]: row[1] for row in result.rows}
            return schema
        except Exception as e:
            print(f"Error fetching table schema: {e}")
            return None

    async def acheck_add_table(self, table_name, ttype: "auth" or "node" or "edge", schema_fetch=True):
        table_exists = await self.acheck_table_exists(table_name)
        print(f">>>>>>{ttype} table {table_name}", table_exists)
        try:
            if not table_exists or table_exists is False:
                table_query = None
                if ttype == "auth":
                    table_query = create_default_auth_table_query(table_name=table_name)
                elif ttype == "node":
                    table_query = create_default_node_table_query(table_name=table_name)
                elif ttype == "edge":
                    table_query = create_default_edge_table_query(table_name=table_name)
                if table_query is not None:
                    print(f"🛠 Creating {ttype} Table: {table_name}")
                    await self.update_db(table_query)

            if schema_fetch:
                schema = await self.afetch_table_schema(table_name)
                return schema
        except Exception as e:
            print(f"Error check add table: {e}")
            await asyncio.sleep(60)
            return await self.acheck_add_table(table_name, ttype, schema_fetch)

    """
    Brille
    Haare
    Parfüm
    Klamotten: Schuhe, Hose, Shirts
    Harddrive
    Essen
    """

    async def update_db(self, query: list or str):
        #print("Run query", query)

        try:
            if isinstance(query, str):
                query = [query]
            request = UpdateDatabaseDdlRequest(database=self.db_path, statements=query)
            operation = await self.adb_client.update_database_ddl(request=request)
            # print("Waiting for operation to complete...")
            response = await operation.result()
            print("Result gathered", response)
            return response
        except Exception as e:
            print("Error update_db", e)

    async def fetch_all_rows_as_dict_async(self, table_name, check_key, check_key_value, select_table_keys) -> list[
        dict]:
        query = self.check_list_value_query(table_name, select_table_keys, check_key, check_key_value)
        response = await self.asnap(query)
        return [dict(row) for row in response.rows]

    async def aadd_col(self, keys: Dict, table, type_from_val=True):
        all_queries = []
        table_schema = await self.afetch_table_schema(table_name=table)
        for k, v in keys.items():
            if k not in table_schema:
                if type_from_val is True:
                    v = self.get_spanner_type(v)
                all_queries.append(
                    self.ddl_add_col(
                        col_name=k,
                        table=table,
                        col_type=v
                    )
                )
        if not len(all_queries):
            print("No new cols for", table)
            return
        print("Insert cols query", all_queries)
        await self.update_db(all_queries)
        #print("All cols added")

    async def update_list(self, table, col_name, new_values, id_insert):
        query = self.update_list_query(col_name, table)
        params = {
            f"{col_name}_add": new_values,
            "id_insert": id_insert
        }

        param_types = {
            f"{col_name}_add": spanner.param_types.Array(spanner.param_types.STRING),
            "id_insert": spanner.param_types.Array(spanner.param_types.FLOAT64)
        }

        request = ExecuteSqlRequest(
            session=self.session.name,
            sql=query,
            params=params,
            param_types=param_types
        )

        await self.aclient.execute_sql(request=request)

    async def async_batch_insert(self, client: SpannerAsyncClient, session_name: str, table: str, rows: list[dict]):
        statements = []

        for row in rows:
            cols = ', '.join(row.keys())
            placeholders = ', '.join([f"@{k}" for k in row.keys()])
            stmt = Statement(
                sql=f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                params=row
            )
            statements.append(stmt)

        request = ExecuteBatchDmlRequest(
            session=session_name,
            transaction={"single_use": {"read_write": {}}},
            statements=statements,
            seqno=1
        )

        response = await client.execute_batch_dml(request=request)
        return response


async def main(spanner_helper=SpannerAsyncHelper()):
    await spanner_helper.acreate_session()
    await spanner_helper.adel_table_batch(
        startswith="GENE",
        endswith="GO",
        table_name=None
    )


if __name__ == "__main__":
    asyncio.run(main())

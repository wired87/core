import asyncio
import csv
import io

from google.cloud import bigquery

from qbrain.a_b_c.bq_agent._bq_core.loader.aloader import ABQHandler
from qbrain.core.app_utils import DB_NAME


class BQController:

    def __init__(self):
        self.abq = ABQHandler(dataset=DB_NAME)

    async def create_table(
            self,
            data:dict
    ):
        # todo create single query for all tables
        print("=========== create-table ===========")
        try:
            table_names = data.get("table_names")
            #attrs = admin_data.get("attrs", None)
            for tn in table_names:
                self.abq.get_create_bq_table(
                    table_name=tn,
                )
            print("created tables")
        except Exception as e:
            print("Error create_table", e)


    async def create_database(self, data: dict, open_src:bool=False):
        print("=========== create-database ===========")
        # todo if open_scr add role allUsers to the ds
        db_name = data["db_name"]
        print("db name:", db_name)
        dataset = bigquery.Dataset(
            self.abq.get_ds_ref()
        )
        try:
            dataset.location = "US"

            # Use the client object to send the dataset configuration to the API
            created_dataset = self.abq.bqclient.create_dataset(dataset)
            print(f"Created dataset {created_dataset.full_dataset_id}")
            """if open_src:
                self.abq.set_ds_open_src(
                    dataset_ref=dataset,
                )"""

        except Exception as e:
            # Handle error, e.g., if the dataset already exists (Conflict) or name is invalid
            print(f"Error creating dataset: {e}")

    async def download_table(self, table_name: str, limit: int = 1000):
        """
        Fetches table admin_data from BigQuery, converts to CSV, and returns as downloadable file.
        """
        query = f"SELECT * FROM `{self.abq.pid}.{self.abq.ds_id}.{table_name}` LIMIT {limit}"
        rows = self.abq.run_query(query, conv_to_dict=True)

        if not rows:
            return {"error": f"No admin_data found in table {table_name}"}

        # CSV in Memory schreiben
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        output.seek(0)

        # StreamingResponse mit CSV Header zurückgeben
        """return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={table_name}.csv"}
        )"""
        return {
            "content": output.getvalue(),
            "filename": f"{table_name}.csv",
            "media_type": "text/csv"
        }


    async def insert(self, table: str, rows: list):
        print("=========== insert ===========")
        self.abq.bq_insert(
            table,
            rows
        )


    async def upsert(self, table: str, rows: list):
        asyncio.create_task(self.abq.async_write_rows(table=table, rows=rows))
        return {"status": "ok", "rows": len(rows)}


    async def query(self, query: str):
        result = self.abq.run_query(query, conv_to_dict=True)
        return {"result": result}


    async def ensure_table(self, table: str, rows: list = None):
        if rows is None:
            rows = []
        ref, schema = self.abq.ensure_table_exists(table, rows)
        return {"table_ref": ref, "schema": [s.to_api_repr() for s in schema]}

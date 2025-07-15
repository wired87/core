"""
failed insert row {...}
 in table cause error: 404 Column not found in table RO: name [resource_type: "spanner.googleapis.com/Column"
resource_name: "name"
]
->
ERROR ADDING COLUMNS: 400 Error parsing Spanner DDL statement: ALTER TABLE BFO ADD COLUMN range STRING(MAX) : Syntax error on line 1, column 28: Encountered 'range' while parsing: identifier
# -> Science Ventures
"""
import json
import os
import re
import time
from typing import List, Dict, Optional

import google

from google.cloud import spanner
import subprocess

from google.cloud.spanner_dbapi import exceptions

from _google import GCP_ID
from gdb_manager.spanner import *

from utils.embedder import embed

"""Creates a database and tables for sample data."""
from google.cloud.spanner_admin_database_v1.types import spanner_database_admin

# Example Usage:
values = [123, 45.67, True, "hello", b"binary", None, [1, 2, 3], {"key": "value"}]


class SpannerGroundZero:
    """Queries & Type validation"""

    def __init__(self):
        self.info_schema = ALL_INFO_SCHEMAS
        self.del_table_batch_size = 100

    def table_schema_query(self, table_name):
        return f"""
          SELECT column_name, spanner_type
          FROM information_schema.columns
          WHERE table_name = '{table_name}'
          ORDER BY ordinal_position
      """

    def table_names_query(self):
        return "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"

    def drop_table_query(self, table_name):
        return f"DROP TABLE {table_name}"

    def drop_graph_query(self, graph_name):
        return f"DROP PROPERTY GRAPH {graph_name}"

    def custom_entries_query(
            self, table_name, check_key=None, check_key_value=None,
            select_table_keys: list or None = None):
        return f"""
            SELECT {','.join(select_table_keys) or '*'} FROM {table_name} 
            WHERE {check_key} = '{check_key_value}'
            """

    def update_list_query(self, col_name, table):
        return f"""
        UPDATE {table}
        SET {col_name} = ARRAY_CONCAT({col_name}, @{col_name}_add)
        WHERE id = @id_insert
        """

    def check_list_value_query(self, table, select_table_keys, array_col, array_check_value):
        return f"""
        SELECT {','.join(select_table_keys) or '*'}
        FROM {table}
        WHERE {array_check_value} IN UNNEST({array_col})
        """

    def ddl_add_col(self, col_name, table="nodes", col_type="STRING(MAX)"):
        """
        Generates a valid ALTER TABLE statement for Google Cloud Spanner.
        Handles STRING, BOOL, and ARRAY<STRING> types correctly.

        Args:
            col_name (str): Name of the column to add.
            table (str): Table name (default "nodes").
            col_type (str): Data type of the column.
        Returns:
            str: Correctly formatted ALTER TABLE SQL statement.
        """
        return f"""
            ALTER TABLE {table} ADD COLUMN `{col_name}` {col_type}
        """

    def add_col_batch_query(self, col_data, table="nodes"):
        batch_query = ""
        for k, v in col_data.items():
            batch_query += f"ADD COLUMN `{k}` {v},"
        batch_query = batch_query[:-1]
        if len(batch_query) == 0:
            return None
        return f"""
            ALTER TABLE {table} 
            {batch_query}
        """

    def get_spanner_type(self, value):
        """
        Determines the Cloud Spanner-compatible data type for a given Python value.
        :param value: The Python value to check.
        :return: The corresponding Cloud Spanner data type.
        """
        if value is None:
            return "STRING(MAX)"
        if isinstance(value, list):
            if len(value) == 0:
                return "ARRAY<STRING(MAX)>"  # Default for empty list
        python_to_spanner = {
            int: "INT64",
            float: "FLOAT64",
            bool: "BOOL",
            str: "STRING(MAX)",
            bytes: "BYTES",
            list: "ARRAY<STRING(MAX)>",
        }
        # todo iwo value as type inserted
        value_type = type(value)

        # Check if it's a list and infer the type
        if isinstance(value, list) and value:
            element_type = self.get_spanner_type(value[0])  # Infer type from first element
            return f"ARRAY<{element_type}>"
        sp_type = python_to_spanner.get(value_type, "STRING(MAX)")
        # print("Spanner type set", sp_type)
        return sp_type


class SpannerCore(SpannerGroundZero):

    def __init__(self, db=None, inid=None):
        """if not local:
            self.client = spanner.Client(project=project_id)
        else:
            os.environ["SPANNER_EMULATOR_HOST"] = "0.0.0.0:9010"
            """

        super().__init__()
        self.client = spanner.Client(project=GCP_ID)
        self.instance = self.client.instance(SP_INSTANCE_ID)

        self.database_id = db or SP_DATABASE_ID
        self.database = self.instance.database(self.database_id)
        self.instance_id = inid or SP_INSTANCE_ID

        self.batch_size = 1000
        self.non_allowed_table_chars = [
            "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "=",
            "[", "]", "{", "}", "\\", "|", ";", ":", "'", '"', "<", ">", ",",
            "?", "/", "`", "~", " ", "\t", "\n", "\r", "\f", "\v"
        ]

    def change_column_type(self, table_name, column_name, new_column_type):
        """
        Changes the data type of a single column in a Google Cloud Spanner table.
        """

        ddl_statement = f"ALTER TABLE {table_name} ALTER COLUMN {column_name} {new_column_type}"

        try:
            operation = self.database.update_ddl([ddl_statement])
            operation.result()  # Wait for the operation to complete
            print(f"Successfully changed column '{column_name}' in table '{table_name}' to type '{new_column_type}'.")
        except Exception as e:
            print(f"Error changing column type: {e}")

    def create_col(self, table, col=None, col_type=None, embed=False):
        alter_query = f"ALTER TABLE {table} ADD COLUMN {col if embed is False else 'embed'} {col_type if embed is False else 'ARRAY<FLOAT64>'}"
        operation = self.database.update_ddl([alter_query])
        operation.result()  # Wait for operation to complete

    def count_table_entries(self, table_name):
        """Returns the total number of rows in the given Spanner table."""
        query = f"SELECT COUNT(*) FROM {table_name}"

        with self.database.snapshot() as snapshot:
            result = list(snapshot.execute_sql(query))

        return result[0][0] if result else 0

    def check_add_table(self, table_name, ttype: "auth" or "node" or "edge", schema_fetch=True):
        if not self.spanner_table_exists(table_name):
            table_query = None
            if ttype == "auth":
                table_query = create_default_auth_table_query(table_name=table_name)
            elif ttype == "node":
                table_query = create_default_node_table_query(table_name=table_name)
            elif ttype == "edge":
                table_query = create_default_edge_table_query(table_name=table_name)
            else:
                table_query = ttype
            if table_query:
                self.create_table(
                    query=table_query
                )
            print(f"Table creation {table_name} successful")
        else:
            print("Table", table_name, "already exists")

        if schema_fetch:
            schema = self.sschema(table_name)
            return schema

    def sp_create_database(self, instance_id, database_id):

        spanner_client = spanner.Client()
        database_admin_api = spanner_client.database_admin_api

        request = spanner_database_admin.CreateDatabaseRequest(
            parent=database_admin_api.instance_path(spanner_client.project, instance_id),
            create_statement=f"CREATE DATABASE `{database_id}`",
        )
        operation = database_admin_api.create_database(request=request)
        print("Waiting for operation to complete...")
        database = operation.result()
        print(
            f"Created database {database_id} on instance {instance_id}".format(
                database.name,
                database_admin_api.instance_path(spanner_client.project, instance_id),
            )
        )

        return database

    def create_table(self, query):
        print(query)
        try:
            operation = self.database.update_ddl(ddl_statements=[query])
            operation.result()
        except Exception as e:
            print(f"ERROR CREATING TABLE: {e}")

    def insert_row(self, table_name, batch_chunk):
        for row in batch_chunk:
            try:
                with self.database.batch() as batch_:
                    batch_.insert(
                        table=table_name,
                        columns=tuple(row.keys()),
                        values=[
                            tuple(row.values())  # Convert to tuples
                        ],
                    )
            except Exception as e:
                print("Insertion failed", e)

    def update_rows(self, table_name, batch_chunk):
        for row in batch_chunk:
            try:
                with self.database.batch() as batch_:
                    batch_.update(
                        table=table_name,
                        columns=tuple(row.keys()),
                        values=[
                            tuple(row.values())  # Convert to tuples
                        ],
                    )
            except Exception as e:
                time.sleep(5)
                try:
                    print("failed update process", e)
                    with self.database.batch() as batch_:
                        batch_.update(
                            table=table_name,
                            columns=tuple(row.keys()),
                            values=[
                                tuple(row.values())
                            ],
                        )
                except Exception as e:
                    print("failed update row in table cause error:", e, "row", row)

    def get_table_ids(self, table_name, id_column_name):
        """Fetch all existing IDs from Spanner."""
        with self.database.snapshot() as snapshot:
            results = snapshot.read(
                table=table_name,
                columns=[id_column_name],
                keyset=google.cloud.spanner.KeySet(all_=True),
            )
            id_list = {row[0] for row in results}  # Use a set for faster lookup
        print(f"Retrieved {len(id_list)} existing IDs from {table_name}")
        return id_list

    def del_table_batch(self, startswith=None, endswith=None, table_name=None):
        if table_name:
            query = self.drop_table_query(table_name)
            operation = self.database.update_ddl([query])
            operation.result()
        else:
            table_names = self.list_spanner_tables()
            for table in table_names:
                if table.startswith(startswith) and table.endswith(endswith):
                    print("Del table", table)
                    query = self.drop_table_query(table)
                    operation = self.database.update_ddl([query])
                    operation.result()

    def delete_col(self, table_name, col_name, schema):
        if col_name in schema:
            alter_query = f"ALTER TABLE {table_name} DROP COLUMN {col_name}"
            operation = self.database.update_ddl([alter_query])
            operation.result()

    def batch_process_rows(self, table_name, id_column_name, rows: List[Dict], adapt: str or None = None):
        """
        Checks if each row exists in Spanner:
          - If missing, adds to `insert_list` for batch insert.
          - If exists, adds to `update_list` for batch update.
        """
        # Step 1: Fetch existing IDs from Spanner
        existing_ids = self.get_table_ids(table_name, id_column_name)

        # Step 2: Split rows into insert and update lists
        insert_list = []
        update_list = []

        for row in rows:
            if not row.get("id"):
                print("SKIP ROW WITHOUT ID", row)
                continue
            row_id = row["id"]  # -> means id not in schema

            if adapt:  # for custom id marker like e
                row_id = adapt + row_id

            if row_id in existing_ids:
                update_list.append(row)
            else:
                insert_list.append(row)

        print(f"Found {len(insert_list)} new rows, {len(update_list)} existing rows of {len(rows)} in {table_name}.")
        self.batch_handle_rows(table_name, insert_list, update_list)
        print(f"Finished processing rows for {table_name}.")

    def add_columns_bulk(self, table_name: str, attrs: dict or list):
        """
        Adds multiple missing columns to a Spanner table in a single batch update.
        :param table_name: The Spanner table name.
        :param new_columns: Dictionary {column_name: column_type}.
        """
        if not attrs:
            print("⚠ No new columns to add.")
            return
        print("Existing attrs.keys()", attrs.keys())

        existing_cols = self.get_table_columns(table_name)  # Fetch existing columns
        print("Existing cols", existing_cols)

        ddl = [
            f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}"
            for col, col_type in attrs.items() if col not in existing_cols
        ]

        if len(ddl):
            try:
                operation = self.database.update_ddl(ddl_statements=ddl)
                operation.result()  # Wait for completion

                print(f"✅ Added missing columns to '{table_name}'")
            except Exception as e:
                print(f"ERROR ADDING COLUMNS: {e}")
                # todo create table
                # self.add_columns_bulk(table_name, attrs)
        else:
            print("NO COLUMNS TO ADD::")

    def batch_handle_rows(self, table_name, rows_to_insert=[], rows_to_update=[]):
        """Batch inserts and updates rows using Spanner batching."""
        for i in range(0, len(rows_to_insert), self.batch_size):
            batch_chunk = rows_to_insert[i:i + self.batch_size]
            try:
                self.insert_row(table_name, batch_chunk)
            except Exception as e:
                print(f"Error while INSERT batch: {e}\n -> insert single\ninti again")
                self.insert_row(table_name, batch_chunk)

        for i in range(0, len(rows_to_update), self.batch_size):
            batch_chunk = rows_to_update[i:i + self.batch_size]
            try:
                self.update_rows(table_name, batch_chunk)
            except Exception as e:
                print(f"Error while UPDATING batch: {e}\n -> insert single")
                self.update_rows(table_name, batch_chunk)

    def update_insert(self, table, rows: List[Dict], schema=None, insert_only=False):
        """Add missing cols, up/ins"""
        """if schema:
            for k, v in rows[0].items():
                self.check_add_cols(
                    key=k,
                    t=table,
                    existing_cols=schema,
                    value=None,
                    col_type=v
                )"""
        for i in range(0, len(rows), self.batch_size):
            batch_chunk = rows[i:i + self.batch_size]
            try:
                success = self.insert_row(table, batch_chunk)
                if not (success or success is False) and insert_only is False:
                    self.update_rows(table, batch_chunk)
            except Exception as e:
                print(f"Error while UPDATING batch: {e}\n -> insert single")
                if insert_only is False:
                    self.update_rows(table, batch_chunk)

    def check_add_cols_batch(self, keys: dict, t):
        for k, v in keys.items():
            try:
                # Try to add the column
                self.database.update_ddl(
                    [
                        self.ddl_add_col(
                            col_name=k,
                            table=t,
                            col_type=v
                        )
                    ]
                ).result()
            except Exception as e:
                if "Duplicate column name" in str(e):
                    print(f"Column {k} already exists")
                else:
                    print(f"Error adding column {k}: {e}")

    def check_add_cols(self, key, t, existing_cols=None, value=None, col_type=None):

        if key not in existing_cols:
            print(f"k {key} not in items")
            try:
                # Try to add the column
                self.database.update_ddl(
                    [
                        self.ddl_add_col(
                            col_name=key,
                            table=t,
                            col_type=self.get_spanner_type(value) if value else col_type
                        )
                    ]
                ).result()
            except Exception as e:
                if "Duplicate column name" in str(e):
                    # todo handling
                    print(f"Column {key} already exists, skipping...")
                else:
                    print(f"Error adding column {key}: {e}")

    def get_all_ids(
            self,
            table_name,
            id_column_name="id",
            batch_size=10000,
            max_retries=1,
            where_not_null_id_col_name=None
    ):

        all_ids = []
        offset = 0
        print("Get ids")
        while True:
            retries = 0
            success = False

            if where_not_null_id_col_name:
                where_not_null_check = f"WHERE {where_not_null_id_col_name} IS NOT NULL"
                base_query = f"""SELECT {id_column_name} FROM {table_name} {where_not_null_check}
                    ORDER BY {id_column_name} """
            else:
                base_query = f"""SELECT {id_column_name} FROM {table_name}
                    ORDER BY {id_column_name} """
            query = base_query + f"LIMIT {batch_size} OFFSET {offset}"

            while retries < max_retries and not success:
                try:
                    with self.database.snapshot() as snapshot:
                        results = snapshot.execute_sql(query)
                        batch_ids = [row[0] for row in results]
                        all_ids.extend(batch_ids)
                        print(f"Fetched {len(batch_ids)} IDs at offset {offset}")
                        success = True
                except Exception as e:
                    retries += 1
                    print(f"Error fetching offset {offset}, retry {retries}/{max_retries}: {e}")
                    # time.sleep(2 ** retries)

            if not success or len(batch_ids) < batch_size:
                break  # Exit if last batch or failed completely

            offset += batch_size

        return all_ids

    def insert_meta(self, key_table: "edge_tables" or "node_tables", table_name):
        with self.database.batch() as batch:
            batch.insert(
                table="metadata",
                columns=(
                    "id",
                    "edge_tables",
                ),
                values=[(
                    self.count_table_entries("metadata") + 1,
                    table_name,
                )],
            )

    def create_meta_table(self, table_name, key_table):
        meta_op = self.database.update_ddl(ddl_statements=[create_metadata_table_query(table_name)])
        meta_op.result()
        self.insert_meta(
            key_table=key_table,
            table_name=table_name
        )

    def create_instance(self, instance_name, region="regional-us-central1", instance_type="free-instance",
                        description="Spanner Graph demo"):

        command = [
            "gcloud", "spanner", "instances", "create", instance_name,
            "--config", region,
            "--instance-type", instance_type,
            "--nodes", 1,
            "--description", description
        ]
        try:
            subprocess.run(command, check=True, text=True)
            print(f"✅ Spanner instance '{instance_name}' created successfully in region '{region}'.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create Spanner instance: {e}")

    def spanner_table_exists(self, table_name):
        """
        Check if a table exists in a Google Cloud Spanner database.

        :param table_name: The table name to check.
        :return: True if the table exists, False otherwise.
        """

        query = f"""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = '{table_name}'
        """

        with self.database.snapshot() as snapshot:
            result = list(snapshot.execute_sql(query))

        return bool(result)  # Returns True if the table exists, False otherwise.

    def list_spanner_tables(self) -> list:
        """Fetch all table names from a Spanner database."""
        try:
            query = self.table_names_query()
            with self.database.snapshot() as snapshot:
                results = list(snapshot.execute_sql(query))
            results = [row[0] for row in results if not row[0] in self.info_schema]
            # print("Filtered Spanner Tables", results)
            return results
        except Exception as e:
            print(f"Error retrieving Spanner tables: {e}")
            return []

    def get_custom_entries(self, table_name, check_key, check_key_value, select_table_keys):
        query = self.custom_entries_query(table_name, check_key, check_key_value, select_table_keys)
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(query)
            return [dict(row) for row in results]

    def update_single_col_entry(self, table, field, value, node_id):
        print("Update row", node_id)
        ddl = f"""
        UPDATE {table}
        SET {field} = {value}
        WHERE id = {node_id};
        """
        with self.database.snapshot() as snapshot:
            snapshot.execute_sql(ddl)

    def filter_edge_src_trgt_from_upper_str(self, input_string):
        parts = input_string.split("_")
        table_refs = [parts[0], parts[-1]]  # first and last part
        if len(table_refs) == 2:
            return table_refs
        print("Error in table filter process", table_refs)

    def edge_graph_query(self, edge_tables: List[str]):
        final_schema = ""
        for edge in edge_tables:
            result = re.findall(r'(?:[A-Z]+_)*[A-Z]+[0-9]*', edge)
            print("Split Edge Table Result", result)

            final_schema += get_edge_table_schema(
                edge_item_name=edge,
                trgt_re_table=result[1],
                src_ref_table=result[0]
            )
        print("Edge Tables Graph creation Query:", final_schema)
        return final_schema

    def create_graph(self, node_tables, edge_tables, graph_name=None):
        edge_table_schema = self.edge_graph_query(edge_tables)

        query = create_graph_query(node_tables, edge_table_schema, graph_name=graph_name)
        try:
            self.create_property_graph2(query)
        except Exception as e:
            print(f"Error creating graph: {e}")

    def create_property_graph2(self, query):
        """Creates a property graph using an UpdateDatabaseDdlRequest instead of a transaction."""

        print("Executing DDL query:", query)

        try:
            spanner_client = spanner.Client()
            instance = spanner_client.instance(self.instance_id)
            database = instance.database(self.database_id)

            operation = database.update_ddl([query])

            print("Waiting for DDL operation to complete...")
            operation.result()  # Wait for the operation to finish
            print("Graph creation completed successfully.")

        except Exception as e:
            print(f"Error creating graph: {e}")

    def get_graph_tables(self, graph_name="BRAINMASTER"):
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                get_graph_query(graph_name)
            )
            print("Graph table names found:", results)
            return results

    def get_graph(self, graph_name):
        with self.database.snapshot() as snapshot:
            try:
                query = f"SELECT GRAPH_NAME FROM INFORMATION_SCHEMA.GRAPHS WHERE GRAPH_NAME = '{graph_name}'"
                results = list(snapshot.execute_sql(query))
                if results:
                    return True
                else:
                    return None
            except Exception as e:
                print("Exception requesting Graph", e)
                return None

    def all_nodes_edges(self) -> dict:
        print("Request graph structure")
        data = {"nodes": [], "edges": []}
        all_tables = self.list_spanner_tables()
        # print("All tables", all_tables)
        data["nodes"], edge_tables = self.filter_table_names(all_tables)
        for et in edge_tables:
            print("Working edge table", et)
            ref = self.filter_edge_src_trgt_from_upper_str(et)
            data["edges"].append(
                {
                    "src": ref[1],
                    "trgt": ref[0],
                }
            )
        # print("All tables", data)
        print("process finished")
        return data

    def filter_table_names(self, table_names):
        print("Filter tables")
        edges = []
        nodes = []
        for table in table_names:
            if re.search(r'[a-z]', table):  # Check for lowercase letters
                edges.append(table)
            else:
                nodes.append(table)

        # print(f"split into nt: {nodes} \net: {edges}")
        print("Tables filtered", nodes, "\n", edges)
        return nodes, edges

    def sschema(self, table_name):
        schema = {}
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(self.table_schema_query(table_name))
            for row in results:
                schema[row[0]] = row[1]
        """
        col: type
        """
        print("Schema fetched:", schema)
        return schema

    def create_or_extend_graph(self):
        """Add all tables if not exist"""
        filtered_g_data = self.all_nodes_edges()
        self.create_graph(filtered_g_data["nodes"], filtered_g_data["edges"])

    def get_graph_tables(self, graph_name: str):
        """
        Retrieves all node and edge table names within a created graph in Spanner.

        :param project_id: GCP Project ID
        :param instance_id: Spanner Instance ID
        :param database_id: Spanner Database ID
        :param graph_name: The name of the graph to retrieve tables from
        :return: Dictionary with lists of node tables and edge tables
        """

        with self.database.snapshot() as snapshot:
            node_query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.GRAPH_TABLES WHERE GRAPH_NAME = '{graph_name}' AND TABLE_TYPE = 'NODE'"
            edge_query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.GRAPH_TABLES WHERE GRAPH_NAME = '{graph_name}' AND TABLE_TYPE = 'EDGE'"

            node_tables = [row[0] for row in snapshot.execute_sql(node_query)]
            edge_tables = [row[0] for row in snapshot.execute_sql(edge_query)]

            return node_tables, edge_tables

    def fetch_table_ids(self):
        all_ids = {}
        all_tables = self.list_spanner_tables()
        for name in all_tables:
            all_ids[name] = self.get_table_columns(name)
        return all_ids

    def get_table_columns(self, table):
        if self.spanner_table_exists(table):
            query = f"""
                        SELECT column_name, spanner_type
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE table_name = '{table}'
                        ORDER BY ordinal_position;
                    """
            with self.database.snapshot() as snapshot:
                results = snapshot.execute_sql(query)
                schema = {row[0]: row[1] for row in results}  # access by index.
                print("Fetched cols from spanner")
            path = rf"C:\Users\wired\OneDrive\Desktop\Projects\bm\data\table_schema\{table}_schema.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Write JSON content asynchronously
            with open(path, "w") as f:
                f.write(json.dumps(schema, indent=2))

            return schema
        return {}


################################################################################################


class SpannerAuth(SpannerCore):

    def __init__(self, db=None):
        super().__init__(db=db)
        self.user_table = "USERS"  # Table where user records are stored

    def get_user(self, email: str) -> Optional[dict]:
        """
        Checks if a user exists in the database by email.

        :param email: User's email to check
        :return: User data dictionary if found, else None
        """
        if not self.spanner_table_exists(self.user_table):
            print("Creating table ", self.user_table)
            self.create_table(
                query=create_default_auth_table_query(table_name=self.user_table)
            )
        query = f"SELECT id, email, password FROM {self.user_table} WHERE email = @email LIMIT 1"
        params = {"email": email}
        param_types = {"email": spanner.param_types.STRING}

        with self.database.snapshot() as snapshot:
            results = list(snapshot.execute_sql(query, params=params, param_types=param_types))

        if results:
            user = {"id": results[0][0], "email": results[0][1], "password": results[0][2]}
            return user
        return None

    def get_user_from_id(self, user_id) -> Optional[dict]:
        """
        Checks if a user exists in the database by email.

        :param email: User's email to check
        :return: User data dictionary if found, else None
        """
        if not self.spanner_table_exists(self.user_table):
            print("Creating table ", self.user_table)
            self.create_table(
                query=create_default_auth_table_query(table_name=self.user_table)
            )
        query = f"SELECT id, email, password FROM {self.user_table} WHERE id = @id LIMIT 1"
        params = {"id": user_id}
        param_types = {"id": spanner.param_types.STRING}

        with self.database.snapshot() as snapshot:
            results = list(snapshot.execute_sql(query, params=params, param_types=param_types))

        if results:
            user = {"id": results[0][0], "email": results[0][1], "password": results[0][2]}
            return user
        return None

    def insert_user(self, user_data: dict) -> bool:
        """
        Inserts a new user into the database.

        :param user_data: Dictionary containing user_id, email, password (hashed)
        :return: True if successful, False otherwise
        """
        try:
            with self.database.batch() as batch:
                batch.insert(
                    table=self.user_table,
                    columns=("id", "email", "password", "created_at"),
                    values=[
                        (user_data["id"], user_data["email"], user_data["password"], user_data["created_at"])
                    ],
                )
            print(f"✅ User {user_data['email']} inserted successfully!")
            return True
        except Exception as e:
            print(f"Error inserting user: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """
        Deletes a user from the database by user_id.

        :param user_id: The unique ID of the user to delete
        :return: True if deletion was successful, False otherwise
        """
        try:
            with self.database.batch() as batch:
                batch.delete(
                    table=self.user_table,
                    keyset=spanner.KeySet(keys=[(user_id,)]),
                )
            print(f"✅ User with ID {user_id} deleted successfully!")
            return True
        except Exception as e:
            print(f"❌ Error deleting user: {e}")
            return False


class SpannerGraphLoader(SpannerCore):
    def __init__(self, db=None):
        super().__init__(db=db)
        self.embedding = False
        self.node_attr_mapping = []
        self.edge_attr_mapping = []

    def check_id_exists(self, keys, node_id, t):
        # print(f"Check for row with id {node_id} result:", row)
        with self.database.snapshot() as snapshot:
            row = snapshot.read(
                table=t,
                columns=(k for k in keys),
                keyset=google.cloud.spanner.KeySet(keys=[(node_id,)]),
            ).one_or_none()
            if row:
                # Convert row to a dictionary
                # print("ROW", row)
                return row

            return None

    def check_id_exists_local(self, node_id, t):
        table = self.all_ids.get(t)
        if table:
            if node_id in table:
                # print("Id already exists")
                return True
        else:
            print("Table not found -> might added while this run")
        return False

    def _modify_keys_insert_cols(self, attrs):
        """
        Modify key fields 
         
        """
        # print("_ATTRS", attrs)
        stuff = {}
        existing_cols = self.get_column_names(table_name=attrs.get('rel'))
        for k, v in attrs.items():
            k_modified = re.sub(r"\.", "_", k).replace("@", "")
            print("key_modified", k_modified)
            stuff[k_modified] = v
            self.check_add_cols(value=v, key=k_modified, t=attrs.get('rel') if attrs.get('rel') else attrs.get('type'),
                                existing_cols=existing_cols)
        # Corrected: Extract columns and values from stuff, and convert to lists.
        return stuff

    def add_g_item(self, attrs):
        """
        Adds or updates a node in a Spanner table.

        Args:
            attrs: A dictionary containing attributes of the node.
        """
        # print("ATTRS", attrs)
        n_type = attrs.get("type")
        src = attrs.get("id")
        if not self.spanner_table_exists(n_type):
            self.create_table(
                query=create_default_node_table_query(table_name=n_type)
            )

        attrs = self._modify_keys_insert_cols(attrs)
        attr_keys = list(attrs.keys())  # Convert dict_keys to a list for indexing
        #
        row_local = self.check_id_exists_local(node_id=src, t=n_type)

        formatted_values = self.handle_values(attrs)

        if row_local is False:
            with self.database.batch() as batch:
                batch.insert(
                    table=n_type,
                    columns=tuple(attr_keys),  # Ensure column names are tuples
                    values=[tuple(formatted_values)],  # Values must be wrapped in a list
                )
            self.add_columns_bulk(table_name=n_type, attrs=attrs)
            # print(f"Added Node: {src}")
        else:
            print(f"Row {src} exists")
            row = self.check_id_exists(node_id=src, keys=attr_keys, t=n_type)

            row = dict(zip(attr_keys, row))
            for k, value in attrs.items():
                if k not in row.keys():
                    self.update_single_col_entry(n_type, k, value, src)  # todo

    def handle_values(self, attrs):
        """
        Converts dictionary values to JSON and keeps lists as arrays for Spanner.

        Args:
            attrs: A dictionary of attributes.

        Returns:
            A list of formatted values.
        """
        values = []
        for v in attrs.values():
            if isinstance(v, dict):
                values.append(json.dumps(v))  # Convert dict to JSON string
            elif isinstance(v, list):
                values.append([json.dumps(lv) if isinstance(lv, dict) else lv for lv in v])
            else:
                values.append(v)
        return values

    def create_edge_table_batch(self, srcl, trgtl, t):
        """Check for all edges"""
        if not self.spanner_table_exists(srcl):
            print("Creating srcl ", srcl)
            self.create_table(
                query=create_default_node_table_query(table_name=srcl)
            )

        if not self.spanner_table_exists(trgtl):
            print("Creating trgtl", trgtl)
            self.create_table(
                query=create_default_node_table_query(table_name=trgtl)
            )

        if not self.spanner_table_exists(t):
            print("Creating t", t)
            self.create_table(
                query=create_default_edge_table_query(src_table_name=srcl, trgt_table_name=trgtl, table_name=t)
            )

    def sadd_edge(self, src, trt, attrs):
        src_layer = attrs.get("src_layer").upper()
        trgt_layer = attrs.get("trgt_layer").upper()

        relationship = attrs.get("rel").lower()

        if relationship:
            t = f"{src_layer}_{relationship}_{trgt_layer}"
            #print("ADD EDGE TO TABLE:", t)

            self.create_edge_table_batch(
                srcl=f"{src_layer}",
                trgtl=f"{trgt_layer}",
                t=t,
            )

            attrs = {"id": src, **(self._modify_keys_insert_cols(attrs))}
            self.insert_row(
                table_name=t,
                data=attrs
            )
            print(f"Edge {src} connected successfully to {trt}.")
        else:
            print("No edge type could be set...")

    def get_column_names(self, table_name: str) -> list:
        """Retrieves all column names from a given table in Spanner."""
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                get_col_names(),
                params={"table_name": table_name},  # Don't force uppercase
                param_types={"table_name": spanner.param_types.STRING}
            )
            columns = [row[0] for row in results]
            # print(f"Col names for {table_name} : {columns}")
            return columns

    def query_graph(self, gql_query):
        """Executes a GQL (Graph Query Language) query."""
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(gql_query)
            return list(results)

    def from_web(self, url):
        """
        https://www.encodeproject.org/report/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.cell_slims=hematopoietic+cell&biosample_ontology.classification=primary+cell&biosample_ontology.cell_slims=myeloid+cell&control_type%21=%2A&status=released&biosample_ontology.system_slims=immune+system&biosample_ontology.system_slims=circulatory+system&biosample_ontology.term_name=naive+thymus-derived+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=T-cell&biosample_ontology.term_name=naive+thymus-derived+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=CD14-positive+monocyte&biosample_ontology.term_name=CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=CD4-positive%2C+alpha-beta+memory+T+cell&biosample_ontology.term_name=CD4-positive%2C+CD25-positive%2C+alpha-beta+regulatory+T+cell&biosample_ontology.term_name=activated+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=CD8-positive%2C+alpha-beta+memory+T+cell&biosample_ontology.term_name=naive+B+cell&biosample_ontology.term_name=CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=stimulated+activated+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=activated+T-cell&biosample_ontology.term_name=activated+naive+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=natural+killer+cell&biosample_ontology.term_name=B+cell&biosample_ontology.term_name=common+myeloid+progenitor%2C+CD34-positive&biosample_ontology.term_name=immature+natural+killer+cell&biosample_ontology.term_name=activated+naive+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=IgD-negative+memory+B+cell&biosample_ontology.term_name=activated+CD8-positive%2C+alpha-beta+memory+T+cell&biosample_ontology.term_name=activated+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=peripheral+blood+mononuclear+cell&biosample_ontology.term_name=stimulated+activated+CD8-positive%2C+alpha-beta+memory+T+cell&biosample_ontology.term_name=activated+CD4-positive%2C+alpha-beta+memory+T+cell&biosample_ontology.term_name=T-helper+17+cell&biosample_ontology.term_name=stimulated+activated+naive+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=T-helper+1+cell&biosample_ontology.term_name=T-helper+2+cell&biosample_ontology.term_name=effector+memory+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=stimulated+activated+naive+B+cell&biosample_ontology.term_name=activated+B+cell&biosample_ontology.term_name=activated+T-helper+1+cell&biosample_ontology.term_name=neutrophil&biosample_ontology.term_name=activated+T-helper+2+cell&biosample_ontology.term_name=activated+T-helper+17+cell&biosample_ontology.term_name=central+memory+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=effector+memory+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=memory+B+cell&biosample_ontology.term_name=stimulated+activated+effector+memory+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=dendritic+cell&biosample_ontology.term_name=suppressor+macrophage&biosample_ontology.term_name=mononuclear+cell&biosample_ontology.term_name=stimulated+activated+memory+B+cell&biosample_ontology.term_name=inflammatory+macrophage&biosample_ontology.term_name=activated+CD4-positive%2C+CD25-positive%2C+alpha-beta+regulatory+T+cell&biosample_ontology.term_name=central+memory+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=activated+effector+memory+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=erythroblast&biosample_ontology.term_name=activated+gamma-delta+T+cell&biosample_ontology.term_name=effector+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=CD1c-positive+myeloid+dendritic+cell&biosample_ontology.term_name=T+follicular+helper+cell&biosample_ontology.term_name=T-helper+22+cell&biosample_ontology.term_name=T-helper+9+cell&biosample_ontology.term_name=activated+T-helper+9+cell&limit=all
        """
        # TODO: Implement web scraping and data extraction logic
        pass

    def list_spanner_graphs(self):
        """Lists all graph names in a Cloud Spanner database.

        Args:
            project_id: The ID of the Google Cloud project.
            instance_id: The ID of the Spanner instance.
            database_id: The ID of the Spanner database.

        Returns:
            A list of graph names, or None if an error occurs.
        """
        try:
            # Execute a query to retrieve graph names from the information schema.
            query = "SELECT GRAPH_NAME FROM INFORMATION_SCHEMA.GRAPHS"
            results = self.database.execute_sql(query)

            graph_names = [row[0] for row in results]
            return graph_names

        except exceptions.GoogleAPICallError as e:
            print(f"Error listing graphs: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def get_g_tables(self):
        """
        Write me a py method that checks if a specific graph exist (name in params)
        if the graph exists, all table names (edges and nodes) gets returned as list format . else: a new graph gets created.
        Now, fetch all specific table names that are present within the Spanner Database. Append each table that isnt present in the graph to it (edge tables end with "_edges" and nodes with "_nodes"
        Use my current codebase to build up on
        """
        # check graph exists
        query = get_graph_table_names_query()
        with self.database.snapshot() as snapshot:
            result = list(snapshot.execute_sql(query))

        if not result:
            print(f"❌ Graph does not exist.")
            return False

        node_tables, edge_tables = result[0]
        return node_tables, edge_tables

    def get_graph(self):
        """Retrieves a specific graph from a Cloud Spanner database by name.

        Returns:
            A dictionary representing the graph, or None if the graph is not found or an error occurs.
            The dictionary will contain the graph metadata.
        """
        try:
            # Execute a query to retrieve the graph metadata.
            query = get_specific_g_info()
            results = self.database.execute_sql(query)

            # Process the results.
            for row in results:
                # Convert the row to a dictionary for easier access.
                graph_data = dict(zip(results.metadata.row_type.field_names, row))
                return graph_data

            # If no rows are returned, the graph was not found.
            return None

        except exceptions.GoogleAPICallError as e:
            print(f"Error retrieving graph: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def add_table_to_graph(self, table_type, table_name):
        """
        Adds a node table to an existing graph in Google Cloud Spanner.

        :param graph_name: The name of the existing graph.
        :param node_table_name: The name of the new node table to add.
        :return: Success message if added.
        """
        # Add node table to the graph using PGQL ALTER GRAPH
        alter_graph_query = add_graph_table(table_type=table_type, table=table_name)

        try:
            with self.database.snapshot() as snapshot:
                snapshot.execute_sql(alter_graph_query)
            print(f"✅ Successfully added node table '{table_name}' to graph")

            # Optional: Update metadata table
            self.insert_meta(f"{table_type.lower()}_tables", table_name)

        except Exception as e:
            print(f"❌ Error adding node table to graph: {e}")
            raise
        print("Finished table-insert-process")

    def check_add_graph_tables(self, node_tables, edge_tables):
        all_spanner_tables = self.list_spanner_tables()
        for sst in all_spanner_tables:
            if not sst in node_tables and sst.endswith("_nodes"):
                self.add_table_to_graph("NODE", table_name=sst)

        for sst in all_spanner_tables:
            if not sst in edge_tables and sst.endswith("_edges"):
                self.add_table_to_graph("EDGE", table_name=sst)


########################################################################################################################


class SpannerRAG(SpannerGraphLoader):

    def __init__(self, graph_name=None, db=None):
        super().__init__(db=db)
        self.graph_name = graph_name

    def spanner_vector_search(self, data, table_name="GO", custom=True, limit=10, select=["id"], embed_row="embed") -> \
    List[Dict]:
        # can handle currently just one field type
        print("data", data)
        print("select", select)

        with self.database.snapshot() as snapshot:
            if custom:
                results = snapshot.execute_sql(
                    f"""
                    SELECT {','.join(select)}, COSINE_DISTANCE(
                    {embed_row}, 
                    @embeds                    
                    ) AS distance
                    FROM {table_name}
                    WHERE {embed_row} IS NOT NULL
                    ORDER BY distance
                    LIMIT @limit;
                    """,
                    params={
                        "embeds": embed(str(data)),
                        "limit": limit
                    },
                    param_types={
                        "data": spanner.param_types.Array(spanner.param_types.FLOAT64),
                        "limit": spanner.param_types.INT64
                    },
                )

            else:
                results = snapshot.execute_sql(
                    f"""
                    SELECT {','.join(select) if select != '*' else select}, COSINE_DISTANCE(
                    {embed_row}, 
                    (SELECT embeddings.values
                    FROM ML.PREDICT(
                    MODEL EmbeddingsModel,
                    (SELECT @data as content)
                    )
                    ) 
                    ) AS distance
                    FROM {table_name}
                    WHERE {embed_row} IS NOT NULL
                    ORDER BY distance
                    LIMIT @limit;
                    """,
                    params={"data": str(data), "limit": limit},
                    param_types={
                        "data": spanner.param_types.STRING if not custom else None,
                        "limit": spanner.param_types.INT64
                    },
                )

            if len(select) == 1:
                formated_results = []

                for k in list(results):
                    kws = {}
                    kws["id"] = k[0]
                    kws["distance"] = k[1]
                    formated_results.append(kws)

                results = formated_results
            else:
                results = list(results)
            print("Results", results)
            return results


if __name__ == "__main__":
    """    sc = SpannerCore()
        all_ids = sc.get_all_ids(
            table_name="GENE", id_column_name="id", batch_size=10000, max_retries=3, where_not_null_id_col_name="start")
    """

    srag = SpannerRAG()
    # srag.spanner_vector_search(data, table_name="GO", custom=True, limit=10, select=["id"], embed_row="embed")

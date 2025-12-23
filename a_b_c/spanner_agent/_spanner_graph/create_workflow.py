from typing import List, Dict

import dotenv
dotenv.load_dotenv()

from a_b_c.spanner_agent import GCP_REGION, GCP_ID
from a_b_c.spanner_agent._spanner_graph.graph_loader import SpannerCore


class SpannerCreator(
    SpannerCore
):
    def __init__(
            self,
    ):
        # Initialize project and region information
        self.region = GCP_REGION
        self.project_id = GCP_ID

        # Default instance config for Europe/West 10 (needs to be a valid Spanner config)
        self.instance_cfg_name = "regional-europe-west10"

        # Use provided/default instance and database IDs for SpannerCore initialization
        SpannerCore.__init__(self)

    def main(
            self,
            instance_id: str,
            instance_display_name: str,
            processing_units: int,
            create_instance: bool,
            create_database: bool,
            database_id: list[str],
            table_definitions: List[Dict] = None
            # [{'name': 'AUTH', 'type': 'auth'}, {'name': 'NODES_A', 'type': 'node'}]
    ):
        """
        Orchestrates the creation of Spanner resources based on boolean flags.

        Args:
            instance_id: The permanent ID for the Spanner instance.
            instance_display_name: The display name for the instance.
            processing_units: Compute capacity (e.g., 100 PUs).
            create_instance: If True, creates the Spanner instance.
            create_database: If True, creates the database within the instance.
            database_id: The ID for the database to create.
            table_definitions: List of dictionaries to define tables.
        """
        if create_instance:
            print("--- Starting Instance Creation ---")
            self.create_spanner_instance(
                project_id=self.project_id,
                instance_id=instance_id,
                config_name=self.instance_cfg_name,
                display_name=instance_display_name,
                processing_units=processing_units
            )
            # Update SpannerCore's instance reference if creation was successful/needed
            self.instance_id = instance_id
            self.instance = self.client.instance(self.instance_id)
            print("----------------------------------")

        if table_definitions:
            print("--- Starting Table Creation ---")
            for table_def in table_definitions:
                table_name = table_def.get('name')
                table_type = table_def.get('ttype')  # Expected: 'auth', 'node', 'edge', or a raw DDL query

                # The check_add_table method is already defined in SpannerCore
                # It handles checking for existence and creating the table based on type/query
                print(f"Checking/Creating table: {table_name} (Type: {table_type})")
                self.check_add_table(
                    table_name=table_name,
                    ttype=table_type,
                    schema_fetch=False  # Skip schema fetch for creation workflow simplicity
                )
            print("----------------------------------")

        print("✅ Creation workflow completed.")


import os

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import List, Optional
from dotenv import load_dotenv


def _process_single_table(
        bq_client: bigquery.Client,
        project_id: str,
        dataset_id: str,
        table_id: str
) -> str:
    """Handles extraction, sheet creation, writing, and setting public permission for one table."""

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{table_ref}`"

    # 2. Extract Data from BigQuery
    try:
        query_job = bq_client.query(query)
        rows = list(query_job.result())
    except Exception as e:
        return f"BigQuery Extraction Failed for {table_id}: {e}"

    # Format data: Header followed by rows
    if not rows:
        data = [["No data found for table."]]
    else:
        header = [field.name for field in query_job.result().schema]
        data = [header] + [list(row.values()) for row in rows]

        csv = pd.DataFrame(
            data=data,
            columns=list(header)
        )
        return csv


def get_convert_bq_table(
        project_id: str,
        dataset_id: str,
        table_id,
        credentials_file: Optional[str] = None
) -> List[str]:
    """
    Retrieves all tables from a BigQuery dataset, extracts data from each,
    loads them into new, public Google Sheets, and returns a list of URLs.
    """

    # 1. Initialize Clients and Authentication
    try:
        # Authentication setup
        if credentials_file and os.path.exists(credentials_file):
            # Pass ALL_SCOPES to ensure token covers BQ metadata and Sheets/Drive
            creds = service_account.Credentials.from_service_account_file(
                credentials_file
            )
        else:
            # Use default application credentials, hoping environment is set up with all required scopes
            creds = None

        # BQ Client initialized with potentially scoped credentials
        bq_client = bigquery.Client(project=project_id, credentials=creds)

    except Exception as e:
        print(f"Error initializing clients: {e}")
        return [f"Initialization Failed: {e}"]

    # 2. Retrieve all table IDs in the dataset
    print(f"Retrieving tables from dataset: {dataset_id}")
    try:
        table_ref = bq_client.dataset(dataset_id, project=project_id).table(table_id)

        # 3. Fetch the table object (metadata, schema, etc.)
        # This executes an API call to get the table's details.
        table = bq_client.get_table(table_ref)
        table_id = table.table_id

        csv = _process_single_table(
            bq_client,
            project_id,
            dataset_id,
            table_id
        )
        return csv
    except Exception as e:
        print(f"Error listing tables in dataset {dataset_id}: {e}")
        return [f"Table Listing Failed: {e}"]





if __name__ == '__main__':
    # Load sensitive information from .env
    load_dotenv()

    # Placeholder values loaded from environment
    MOCK_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "aixr-401704")
    MOCK_DATASET_ID = os.environ.get("BQ_DATASET_ID", "TheBuilder")
    MOCK_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    print("--- Starting BigQuery to Sheets Transfer ---")
    print(f"Targeting Project: {MOCK_PROJECT_ID}, Dataset: {MOCK_DATASET_ID}")

    result_urls = get_convert_bq_table(
        project_id=MOCK_PROJECT_ID,
        dataset_id=MOCK_DATASET_ID,
        credentials_file=MOCK_CREDENTIALS_PATH
    )

    print("\n--- Final Result List (URLs or Errors) ---")
    for url_or_error in result_urls:
        print(url_or_error)
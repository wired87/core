import os
from typing import List, Optional
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
import pandas as pd


def fetch_dataset_to_csv_list(
        project_id: str,
        dataset_id: str,
        credentials_file: Optional[str] = None
) -> List[str]:
    """
    Fetches the content of all tables in a BigQuery dataset and returns
    each table as a complete CSV-formatted string.

    Args:
        project_id: GCP project ID.
        dataset_id: BigQuery dataset ID.
        credentials_file: Optional path to the Service Account JSON key.

    Returns:
        List[str]: A list of CSV strings, one for each table.
    """
    csv_outputs = []

    # 1. Initialize BigQuery Client
    try:
        if credentials_file and os.path.exists(credentials_file):
            creds = service_account.Credentials.from_service_account_file(
                credentials_file
            )
        else:
            creds = None

        bq_client = bigquery.Client(project=project_id, credentials=creds)

    except Exception as e:
        print(f"ERROR: Initialization Failed: {e}")
        return [f"Initialization Error: {e}"]

    # 2. Retrieve all table IDs
    try:
        dataset_ref = bq_client.dataset(dataset_id, project=project_id)
        tables = list(bq_client.list_tables(dataset_ref))
    except Exception as e:
        print(f"ERROR: Table Listing Failed: {e}")
        return [f"Table Listing Error: {e}"]

    # 3. Loop through tables and fetch CSV data
    for table in tables:
        table_ref = f"{project_id}.{dataset_id}.{table.table_id}"
        query = f"SELECT * FROM `{table_ref}`"

        try:
            # Run query and convert results directly to a Pandas DataFrame
            df = bq_client.query(query).to_dataframe()

            # Convert DataFrame to CSV string (includes header)
            csv_string = df.to_csv(index=False)
            csv_outputs.append(csv_string)
            print(f"✅ Fetched and formatted CSV for: {table.table_id}")

        except Exception as e:
            error_msg = f"Fetch Error for {table.table_id}: {e}"
            print(f"❌ {error_msg}")
            csv_outputs.append(f"ERROR: {error_msg}")

    return csv_outputs


if __name__ == '__main__':
    load_dotenv()

    # --- Example Usage (Using Public Dataset) ---
    MOCK_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "bigquery-public-data")
    MOCK_DATASET_ID = os.environ.get("BQ_DATASET_ID", "QCOMPS")
    MOCK_CREDENTIALS_PATH = r"C:\Users\bestb\PycharmProjects\BestBrain\auth\credentials.json" if os.name == "nt" else "auth/credentials.json"

    if MOCK_PROJECT_ID == "bigquery-public-data":
        print("NOTE: Using public BQ data. No credentials required for public data access.")

    print("\n--- Starting Data Fetch ---")

    all_csv_data = fetch_dataset_to_csv_list(
        project_id=MOCK_PROJECT_ID,
        dataset_id=MOCK_DATASET_ID,
        credentials_file=MOCK_CREDENTIALS_PATH
    )

    print("\n--- Final CSV Data List (Truncated Output) ---")
    for i, csv_content in enumerate(all_csv_data):
        if csv_content.startswith("ERROR"):
            print(f"Item {i + 1}: {csv_content}")
        else:
            # Print only the first 200 characters of the CSV content
            print(f"Item {i + 1} (Table CSV): {csv_content[:200]}...")
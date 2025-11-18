import os
from google.cloud import bigquery
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Define a comprehensive list of scopes required for all operations
ALL_SCOPES = [
    # CRITICAL FIX: Changed from bigquery.readonly to full bigquery scope
    # to allow the client to run query jobs (JobService.InsertJob).
    "https://www.googleapis.com/auth/bigquery",
    # Required for creating/editing sheets
    "https://www.googleapis.com/auth/spreadsheets",
    # Required for setting public permissions on the file
    "https://www.googleapis.com/auth/drive"
]


def _process_single_table(
        bq_client: bigquery.Client,
        sheets_service: Any,
        drive_service: Any,
        project_id: str,
        dataset_id: str,
        table_id: str
) -> str:
    """Handles extraction, sheet creation, writing, and setting public permission for one table."""

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{table_ref}` LIMIT 10000"

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

    # 3. Create New Google Sheet
    spreadsheet_body = {
        'properties': {'title': f"BQ Export: {dataset_id}.{table_id} (Generated)"},
        'sheets': [{'properties': {'title': 'Data'}}]
    }

    try:
        spreadsheet = sheets_service.spreadsheets().create(body=spreadsheet_body).execute()
        spreadsheet_id = spreadsheet.get('spreadsheetId')
    except HttpError as e:
        return f"Sheet Creation Failed for {table_id}: {e.content}"

    # 4. Write Data to Sheet (A1)
    try:
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='Data!A1',
            valueInputOption='USER_ENTERED',
            body={'values': data}
        ).execute()
    except HttpError as e:
        return f"Sheet Write Failed for {table_id}: {e.content}"

    # 5. Set Sharing Permission to Public (Reader)
    try:
        permission_body = {'type': 'anyone', 'role': 'reader'}
        drive_service.permissions().create(
            fileId=spreadsheet_id,
            body=permission_body,
            fields='id'
        ).execute()

        return spreadsheet.get('spreadsheetUrl')

    except HttpError as e:
        return f"Sharing Failed for {table_id}: {e.content}"


def transfer_bq_to_public_sheet(
        project_id: str,
        dataset_id: str,
        credentials_file: Optional[str] = None
) -> List[str]:
    """
    Retrieves all tables from a BigQuery dataset, extracts data from each,
    loads them into new, public Google Sheets, and returns a list of URLs.
    """

    result_urls = []

    # 1. Initialize Clients and Authentication
    try:
        # Authentication setup
        if credentials_file and os.path.exists(credentials_file):
            # Pass ALL_SCOPES to ensure token covers BQ metadata and Sheets/Drive
            creds = service_account.Credentials.from_service_account_file(
                credentials_file, scopes=ALL_SCOPES
            )
        else:
            # Use default application credentials, hoping environment is set up with all required scopes
            creds = None

        # BQ Client initialized with potentially scoped credentials
        bq_client = bigquery.Client(project=project_id, credentials=creds)
        sheets_service = build('sheets', 'v4', credentials=creds)
        drive_service = build('drive', 'v3', credentials=creds)

    except Exception as e:
        print(f"Error initializing clients: {e}")
        return [f"Initialization Failed: {e}"]

    # 2. Retrieve all table IDs in the dataset
    print(f"Retrieving tables from dataset: {dataset_id}")
    try:
        dataset_ref = bq_client.dataset(dataset_id, project=project_id)
        # The list_tables call requires the 'bigquery.readonly' scope (or broader)
        tables = list(bq_client.list_tables(dataset_ref))
    except Exception as e:
        print(f"Error listing tables in dataset {dataset_id}: {e}")
        return [f"Table Listing Failed: {e}"]

    # 3. Loop through each table and process
    if not tables:
        return [f"No tables found in dataset {dataset_id}."]

    for table in tables:
        table_id = table.table_id

        url = _process_single_table(
            bq_client,
            sheets_service,
            drive_service,
            project_id,
            dataset_id,
            table_id
        )

        # Log result to console and collect URL or error message
        if url.startswith("http"):
            print(f"✅ Success: Table {table_id} URL: {url}")
            result_urls.append(url)
        else:
            print(f"❌ Failure for Table {table_id}: {url}")
            result_urls.append(f"ERROR: {table_id} -> {url}")

    return result_urls


if __name__ == '__main__':
    # Load sensitive information from .env
    load_dotenv()

    # Placeholder values loaded from environment
    MOCK_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "aixr-401704")
    MOCK_DATASET_ID = os.environ.get("BQ_DATASET_ID", "TheBuilder")
    MOCK_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    print("--- Starting BigQuery to Sheets Transfer ---")
    print(f"Targeting Project: {MOCK_PROJECT_ID}, Dataset: {MOCK_DATASET_ID}")

    result_urls = transfer_bq_to_public_sheet(
        project_id=MOCK_PROJECT_ID,
        dataset_id=MOCK_DATASET_ID,
        credentials_file=MOCK_CREDENTIALS_PATH
    )

    print("\n--- Final Result List (URLs or Errors) ---")
    for url_or_error in result_urls:
        print(url_or_error)
import os

from google.auth.credentials import Credentials
from google.oauth2 import service_account


def load_service_account_credentials(file_path: str) -> Credentials:
    """Loads a Service Account file into the universal Credentials object."""
    if os.path.exists(file_path):
        return service_account.Credentials.from_service_account_file(
            file_path
        )
    else:
        print("no creds found under", file_path)
    return None
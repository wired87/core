import os

from google.auth.credentials import Credentials
from google.oauth2 import service_account

GACF=r"C:\Users\bestb\PycharmProjects\core\auth\credentials.json" if os.name == "nt" else "auth/credentials.json"

def load_service_account_credentials() -> Credentials:
    """Loads a Service Account file into the universal Credentials object."""
    if os.path.exists(GACF):
        return service_account.Credentials.from_service_account_file(
            GACF
        )
    else:
        print("no creds found under", GACF)
    return None
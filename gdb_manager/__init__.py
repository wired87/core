import os

from firebase_admin import credentials
from google.oauth2 import service_account

from utils.get_creds import get_creds

def set_creds():
    g_creds_path_exists = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    fb_creds = os.environ.get("FIREBASE_CREDENTIALS")
    if os.name != "nt" and not g_creds_path_exists and fb_creds is None:
        creds = get_creds(
            types=[
                "g_creds",
                "fb_creds"
            ]
        )
        if "g_creds" in creds and "fb_creds" in creds:
            os.environ["FIREBASE_CREDENTIALS"] = credentials.Certificate(creds["fb_creds"])

            os.environ.setdefault(
                "GOOGLE_APPLICATION_CREDENTIALS",
                service_account.Credentials.from_service_account_info(
                    creds["g_creds"]
                )
            )
            print("Creds successfully set")
        else:
            print("Creds couldnt be set.")
            exit(1)


if __name__== "__main__":
    print("Set DB Creds")
    set_creds()
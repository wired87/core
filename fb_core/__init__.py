import os

from dotenv import load_dotenv
load_dotenv()
FIREBASE_CREDS=r"C:\Users\bestb\PycharmProjects\BestBrain\auth\firebase_creds.json" if os.name == "nt" else "auth/firebase_creds.json"
APP_ROOT=os.getenv("APP_ROOT")
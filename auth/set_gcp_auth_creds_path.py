import os


def set_gcp_auth_path():
    if os.name == "nt":
        GOOGLE_APPLICATION_CREDENTIALS = r"C:\Users\bestb\PycharmProjects\BestBrain\auth\credentials.json"
    else:
        GOOGLE_APPLICATION_CREDENTIALS = "/home/derbenedikt_sterra/BestBrain/_google/g_auth/aixr-401704-59fb7f12485c.json"
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GOOGLE_APPLICATION_CREDENTIALS)

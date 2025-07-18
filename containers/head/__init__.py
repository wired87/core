import os

from fastapi import FastAPI

from bm.settings import TEST_ENV_ID, TEST_USER_ID

app = FastAPI()

USER_ID = os.environ.get("USER_ID", TEST_USER_ID)
ENV_ID = os.environ.get("ENV_ID", TEST_ENV_ID)



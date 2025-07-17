import os

from fastapi import FastAPI
from ray import serve

from containers.head.main import HeadDepl

app = FastAPI()

USER_ID = os.environ.get("USER_ID")
ENV_ID = os.environ.get("ENV_ID")

if __name__ == "__main__":
    print("Start Server...")
    serve.run(
        HeadDepl.options(
            name=ENV_ID
        ).bind(),
        route_prefix=f"/"
    )
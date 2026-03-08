import httpx
import yaml
from pydantic import create_model


def start_server():
    from fastapi import FastAPI
    global app
    app = FastAPI()

    with open("cfg.yaml") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        print("No config found...")

    for item in cfg:
        case=item["case"]
        path = item["path"]
        description = item["description"]
        req_struct = item["req_struct"]

        model = create_model(f"{case}Model", **req_struct)

        async def _handler(payload:model):
            return httpx.post(url=f"http://localhost:8000/{path}", json=payload.dict())




        app.add_api_route(
            path,
            endpoint,
            methods=["POST"],
            summary=description,  # Wichtig für Gemini!
            description=description
        )




if __name__ == "__main__":
    start_server()
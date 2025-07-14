from ray import serve

from .server import ServerWorker

if __name__ == "__main__":
    print("Start Server...")
    serve.run(
        ServerWorker.bind(),
        route_prefix="/"
    )
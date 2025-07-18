from ray import serve

from containers.head.main import HeadDepl

if __name__ == "__main__":
    print("Sstart Server...")
    serve.run(
        HeadDepl.bind(),
        route_prefix="/"
    )
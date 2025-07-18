import os


#app = FastAPI()

USER_ID = os.environ.get("USER_ID")
ENV_ID = os.environ.get("ENV_ID")
QFN_ID = os.environ.get("QFN_ID")

"""if __name__ == "__main__":
    print(f"Start {QFN_ID}...")
    ray.start(
        QFNWorker.options(name=QFN_ID).remote()
    )
"""
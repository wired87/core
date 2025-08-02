import subprocess

from ggoogle.storage.storage import GBucket
from gnn.main import SRC_PATH

def get_all_bucket():
    bucket= GBucket("bestbrain")
    blobs = bucket.list_bucket_objects()
    for b in blobs:
        print("Workinig", b.name)
        if b.name.endswith(".json") and b.name.startswith("graph_model/train_data/"):
            print("Move file", b.name)
            subprocess.run(f"gsutil mv gs://bestbrain/{b.name} gs://{bucket.bucket_name}/{SRC_PATH['paths']['train_data']['bucket']}/{b.name.split('/')[-1]}", shell=True)



if __name__ == "__main__":

    pass

import os
import json

import networkx as nx
import requests
from google.cloud import storage

import dotenv
from tqdm import tqdm

from utils.file import asave_json_content
from collections import defaultdict

from _google import GCP_ID
from _google.gdb_manager.storage import MAIN_BUCKET

dotenv.load_dotenv()


class GBucket:
    def __init__(self, bucket_name=None):
        self.client = storage.Client(project=GCP_ID)
        self.bucket = self.client.bucket(bucket_name or MAIN_BUCKET)
        self.bucket_name = bucket_name or MAIN_BUCKET

    async def download_folder_content_or_single(self,prefix, name=None):
        """Downloads content from a folder or a single file from cloud storage."""
        try:
            blobs = await self.gather_folder_blobs(prefix)
            if not blobs:
                print("No blobs found to download.")
                return False

            for blob in blobs:
                blob.download_to_filename(blob.name)  # Adjust based on your directory structure
                if name == blob.name:
                    return
            print("Download completed.")

        except Exception as e:
            print("Error during download:", e)
            return False

    def upload_mount_vol(self, received_file, dest):
        with open(dest, "wb") as f:
            for chunk in received_file.chunks():
                f.write(chunk)


    async def save_process(self, file_name, content, layer=None):

        if layer is None:
            dest_local = os.path.join(self.local_dest_base, file_name)
            dest_bucket = self.bucket_dest_base + file_name
        else:
            dest_local = os.path.join(self.local_dest_base, layer, file_name)
            dest_bucket = f"{self.bucket_dest_base}{layer}/{file_name}"

        await self.asave_ckpt_local(dest_local, content)
        self.bucket.upload_from_str(dest_path=dest_bucket, content=content)

    async def get_process(self, request_path):
        json_file: bool = request_path.endswith('.json')
        file_name = request_path.split("/")[-1]
        save_local = os.path.join(self.local_dest_base, file_name)
        try:

            if os.path.exists(request_path):
                print("Fetch content local")
                content = await self.aread_content(request_path, j=json_file)
            elif list(self.bucket.bucket.list_blobs(prefix=request_path, max_results=1)):
                print("Fetch file from bucket")
                content = json.loads(self.bucket.download_blob(request_path))
                await self.asave_ckpt_local(path=save_local, content=content)

            else:
                print("Fetch from web")
                content = await self.download_json_content(url=request_path, j=json_file, save=file_name)
            return content
        except Exception as e:
            print("not file found:", e)
        return None

    async def save_layer_ckpt(self, bucket_path, loal_path, data=None, graph_type=None):
        """
        Save the graph as a JSON file and upload it to the bucket.
        """
        print("upload graph to bucket")

        if graph_type in ["protein", "gene"]:
            print("Graph needs to be split since it's too heavy")


        # Convert graph to node-link format
        data = nx.node_link_data(data)
        data["is_multigraph"] = False

        # 🔹 Ensure `loal_path` is a string before passing it
        loal_path_str = str(loal_path)

        # Upload JSON data
        await self.bucket.upload_json_to_folder(
            json_content=data,
            bucket_path=bucket_path,
            local_path=loal_path_str  # 🔹 Fix: Convert Path to String
        )


    async def gather_folder_blobs(self, prefix):
        try:
            jsonl_blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return jsonl_blobs
        except Exception as e:
            print("Failed download:", e)


    def extract_gcs_train_tree(self, bucket_name, prefix=""):
        paths_list = []
        # Initialize GCS client
        bucket = self.bucket.client.bucket(bucket_name)

        def crawl_structure(prefix, paths_list):
            blobs = list(bucket.list_blobs(prefix=prefix))
            #print("len bl", len(blobs))
            for blob in blobs:
                # Get full file path
                file_path = blob.name
                #print("blob_name", file_path)

                # Extract folder path and filename
                if file_path.endswith("/"):
                    prefix = blob.name
                    crawl_structure(prefix, file_path)
                else:
                    #print("Found file:", blob.name)
                    paths_list.append(blob.name)

        crawl_structure(prefix, paths_list)
        print("Extracted files:", paths_list)
        return paths_list



    async def download_single_file(self, prefix, file_name, dest_path=None, download_as_string=False):
        try:

            print(f"Download file {prefix}{file_name} from bucket {self.bucket_name} to dest {dest_path}")
            jsonl_blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            for jLblob in jsonl_blobs:
                if jLblob.name == f"{prefix}{file_name}":
                    if download_as_string:
                        return jLblob.download_as_string()
                    dirn=os.path.dirname(dest_path)
                    os.makedirs(dirn, exist_ok=True)
                    save_path = os.path.join(f"{dest_path}")
                    jLblob.download_to_filename(save_path)
                    print(f"Saved file")
                    return True
        except Exception as e:
            print("Failed download:", e)
        return False





    def upload_file(self, local_file_path, remote_path):
        """Upload a local file to a specific path in the bucket."""
        blob = self.client.bucket(self.bucket_name).blob(remote_path)
        blob.upload_from_filename(local_file_path)
        print(f"Uploaded {local_file_path} to {remote_path}.")

    def get_create(self):
        try:
            bucket = self.client.create_bucket(self.bucket_name)
            print(f"Bucket {self.bucket_name} created.")
            return bucket
        except Exception as e:
            print("BUCKET COULD NOT BE CREATED...", e)

    def list_bucket_objects(self):
        try:
            blobs = self.client.list_blobs(self.bucket_name)
            print(f"\nBucket {self.bucket_name}:")
            return blobs
        except Exception as e:
            print("Could not get the blobs...", e)
            return []

    def get_jsonl_file_names(self, blobs):
        jsonl_file_names = []
        for blob in blobs:
            print("Working file", blob.name)
            if blob.name == "jsonl/":
                jsonl_blobs = self.client.list_blobs(self.bucket_name, prefix=blob.name)
                for jLblob in jsonl_blobs:
                    if jLblob.name.endswith(".jsonl"):
                        jsonl_file_names.append(jLblob.name)
        return jsonl_file_names

    def gather_text_content(self, blobs):
        json_file_contents = {}
        existing_jsonl_file_names = self.get_jsonl_file_names(blobs)
        for blob in blobs:
            if blob.name in existing_jsonl_file_names:
                print("File already converted to JSONL:", blob.name)
            elif blob.name.endswith(".json"):
                content = blob.download_as_text()
                print("Content downloaded for:", blob.name)
                try:
                    json_file_contents[blob.name] = json.loads(content)  # Parse JSON
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for {blob.name}: {e}")
        return json_file_contents

    def get_single_blob(self, file_name, blobs):
        return next((blob for blob in blobs if blob.name == file_name), None)


    def upload_bucket(self, dest_path, src_path):
        blob = self.bucket.blob(dest_path)
        print("Uploading file to GS")
        blob.upload_from_filename(src_path)

    def upload_from_str(self, dest_path, content):
        blob = self.bucket.blob(dest_path)
        print("Uploading content to GS")
        blob.upload_from_string(content)


    async def upload_json_to_folder(self, bucket_path, local_path=None, json_content=None, file_name=None):
        """
        Upload JSON content to a specific folder in the bucket.
        Args:
            json_content (dict): JSON data to upload.
            folder_path (str): Folder path in the bucket (e.g., "my-folder/").
            file_name (str): Name of the file to create in the bucket.
        """
        # Ensure the folder path ends with a slash
        try:
            if json_content and local_path:
                await asave_json_content(path=local_path, content=json.dumps(json_content, indent=2))
                print(f"Upload content from {local_path} to", bucket_path)
            await self.aupload_json_to_folder(json_content=json.dumps(json_content), folder_path=bucket_path, file_name=file_name)
        except Exception as e:
            print("Error uploading", e)

    async def aupload_json_to_folder(self, json_content, folder_path, file_name=None):
        """
        Upload JSON content to a specific folder in the bucket.

        Args:
            json_content (dict): JSON data to upload.
            folder_path (str): Folder path in the bucket (e.g., "my-folder/").
            file_name (str): Name of the file to create in the bucket.
        """
        # Ensure the folder path ends with a slash
        try:
            # Construct the full remote path
            if file_name:
                if not folder_path.endswith("/"):
                    folder_path += "/"
                remote_path = folder_path + file_name
            else:
                remote_path = folder_path
            print("remote_path", remote_path)
            # Get the bucket and blob

            client = storage.Client()
            bucket = client.bucket("bestbrain")
            blob = bucket.blob(remote_path)

            blob.upload_from_string(json_content, "application/json")
            print(f"Uploaded JSON content to {remote_path}")
        except Exception as e:
            print("Error uploading", e)


    def download_blob(self, bucket_path, t="str"):
        blob = self.bucket.blob(bucket_path)
        if t=="str":
            return blob.download_as_text()
        #blob.download_to_filename(dest_path)


    def web_bulk_upload(self, urls, dest):
        [self.bucket.blob(dest).upload_from_string(requests.get(url)) for url in urls]



    def upload_from_url(self, file_url, blob_name):
        """ Uploads a file from URL to Google Cloud Storage with progress tracking """
        print("Uploading:", file_url)
        try:
            blob = self.bucket.blob(blob_name)
            with blob.open("wb") as f, requests.get(file_url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                with tqdm(total=total_size, desc=f"Uploading {blob_name}", unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):  # 8 KB chunks
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return f"gs://{self.bucket_name}/{blob_name}"
        except requests.RequestException as e:
            raise Exception(f"Failed to download file: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to upload to GCS: {str(e)}")


    def get_folders_with_files(self, prefix):
        folder_tree = defaultdict(set)

        # Iterate over all blobs (files and folders) in the bucket
        for blob in self.client.list_blobs(self.bucket_name, prefix=prefix):
            full_path = blob.name  # Example: "folder1/folder2/folder3/file.txt"
            path_parts = full_path.split("/")  # Split into folder structure

            # Ensure it's a file (not an empty folder marker)
            if len(path_parts) > 1:
                for i in range(len(path_parts) - 1):
                    folder_tree["/".join(path_parts[:i + 1])].add(full_path)  # Map parent folders to files

        # Find deepest folders that contain files
        deepest_folders = set()
        for folder, files in folder_tree.items():
            subfolders = [f for f in folder_tree if f.startswith(folder + "/") and f != folder]
            if not subfolders:  # No deeper subfolders, meaning it's the deepest one with files
                deepest_folders.add(folder)

        return sorted(deepest_folders)  # Return a sorted list for consistency



def convert_to_graph(file_path: str) -> nx.Graph:
    """Converts a file into a networkx graph, supporting multiple formats."""
    _, ext = os.path.splitext(file_path)

    try:
        if ext == ".json":
            return nx.read_graphml(file_path)  # Assuming JSON is in GraphML format
        elif ext == ".csv":
            return nx.read_edgelist(file_path, delimiter=",")
        elif ext == ".edgelist":
            return nx.read_edgelist(file_path)
        else:
            print("Unsupported file format.")
            return None
    except Exception as e:
        print(f"Error converting file to graph: {e}")
        return None



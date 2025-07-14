"""
https://granddb.s3.us-east-2.amazonaws.com/data/EGRET_GM12878.csv
https://granddb.s3.amazonaws.com/tissues/networks/lioness/Brain_Cerebellum_AllSamples.csv
https://granddb.s3.amazonaws.com/tissues/networks/lioness/Brain_Other_AllSamples.csv
https://granddb.s3.amazonaws.com/tissues/networks/lioness/Brain_Basal_Ganglia_AllSamples.csv


astrocyte ENCODE
https://www.encodeproject.org/files/ENCFF457ZXS/@@download/ENCFF457ZXS.tar.gz
https://www.encodeproject.org/files/ENCFF342BJP/@@download/ENCFF342BJP.bam
https://www.encodeproject.org/files/ENCFF620TIV/@@download/ENCFF620TIV.bam


nih
dbgap snp - https://ftp.ncbi.nlm.nih.gov/dbgap/studies/

"""

import asyncio
import os
import re

from django.http import JsonResponse
from rest_framework import serializers
from rest_framework.views import APIView

from _google.gdb_manager.bq.dj.views.gs2csv2bq import extract_urls
from _google.gdb_manager.storage.storage import GBucket

STORAGE_HANDLER = GBucket()


class S(serializers.Serializer):
    database = serializers.CharField(
        default="database",
        label="database",
    )
    layer = serializers.CharField(
        default="layer",
        label="layer",
    )



class WebToCloudStorage(APIView):
    serializer_class = S

    """
    Handles uploading files to cloud storage and processing ENCODE datasets.
    """

    def upload_from_url_async(self, file_url, blob_name):
        """Threaded method to upload a file from URL to Google Cloud Storage with tqdm progress updates."""
        try:
            gcs_url = STORAGE_HANDLER.upload_from_url(file_url, blob_name)
            return gcs_url
        except Exception as e:
            print(f"Failed to upload {file_url}: {e}")
            return None

    def sanitize_filename(self, filename):
        """ Remove invalid characters from filenames. """
        return re.sub(r'[<>:"/\\|?*]', '_', filename)  # Replaces invalid characters with "_"

    def get_file_name(self, url):
        if "type" in url:
            file_name = url
        else:
            file_name = url.split('/')[-1]
        return file_name


    def progress_bar(self, urls, layer, part, sub_dir=None):
        """Handles the progress bar for uploading multiple files in parallel."""
        for url in urls:
            self.upload_from_url_async(
                file_url=url,
                blob_name=f"train_data/{layer}/{part}/{sub_dir}/{self.get_file_name(url)}"

            )


    def upload_info_file(self, info, storage_handler, layer, part, sub_dir=None):
        """Uploads JSON metadata file to storage."""
        print(f"Uploading metadata for {sub_dir}")
        asyncio.run(
            storage_handler.aupload_json_to_folder(
                json_content=info,
                folder_path=f"train_data/{layer}/{part}/{sub_dir}/info.json")
        )


    def process_files(self, local, layer, part, uploaded_files, request, prefix=""):
        print("process_files")

        if local and os.path.exists(local):
            if os.path.isdir(local):
                print("Working on directory:", local)
                for path in os.listdir(local):
                    full_path = os.path.join(local, path)
                    folder_name = os.path.basename(full_path)
                    print("basename", folder_name)
                    self.process_files(full_path, layer, part, uploaded_files, request, prefix=folder_name)
                return
            elif os.path.isfile(local):
                print("Working on file:", local)
                try:
                    with open(local, "r", encoding="utf-8") as f:
                        data = f.read()
                except Exception as e:
                    print(f"Error reading file {local}: {e}")
                    return
            else:
                print("Path does not exist:", local)
                return

        else:
            data = request.data.get("data", "")

        if not data.strip():
            print("No data found in file or request.")
            return

        urls = extract_urls(data)
        print(f"Extracted URLs: {urls}")

        if urls:
            self.progress_bar(urls, layer, part, sub_dir=prefix)
        else:
            print("No valid URLs extracted.")

    def post(self, request):


        """Handles the POST request to upload files and process ENCODE datasets."""
        uploaded_files = []
        data_path = request.data.get("data_path") or rf"C:\Users\wired\OneDrive\Desktop\Projects\brainmaster_backend2\data\encode\data"
        database = request.data.get("database") or "encode"
        layer = request.data.get("layer") or "functional_genomics"

        print(f"Processing Layer: {database}, Part: {layer}")

        self.process_files(data_path, database, layer, uploaded_files, request)

        return JsonResponse({
            "message": "Download & upload complete",
            "uploaded_files": uploaded_files
        })




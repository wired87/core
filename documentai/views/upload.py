import os
from datetime import date

import requests
from rest_framework.response import Response
from rest_framework.views import APIView

from google.cloud import aiplatform


from bm.settings import REQUEST_URL

from rest_framework import serializers

from _google import GCP_ID
from documentai.proces_doc import process_document_sample
#from utils.embedder import embed



# Allowed file types and size constraints
ALLOWED_EXTENSIONS = {"jpeg", "jpg", "png", "bmp", "pdf", "tiff", "tif", "gif"}
MAX_FILE_SIZE_MB = 20
MAX_FILES = 15
# Initialize Vertex AI
aiplatform.init(project=GCP_ID, location="us-central1")


class DocumentUploadSerializer(serializers.Serializer):
    user_id = serializers.CharField(
        default="user_id",
        label="User ID",
        help_text="Unique user identifier"
    )
    files = serializers.FileField(
        default="files",
        label="files",
        help_text="List of files to upload"
    )

class DocumentUploadView(APIView):
    serializer_class = DocumentUploadSerializer

    """
    Todo: better annotation incl. images sep col,...
    """
    """def handle_file_insertion_bq(self, doc_data, table_id):

        if isinstance(doc_data, list) and len(doc_data) > 0:
            example_row = doc_data[0]
            schema = bq_auth_handler.schema_from_dict(example_row, embed=True)
            print("schema", schema)
            print(f"DEBUG: Inserting data into table: {table_id}")
            bq_auth_handler.get_create_table(table_id, schema=schema)  # create table once.
            for item in doc_data:
                print(f"Inserting row")
                bq_auth_handler.bq_insert(table_id, row=item)
                print(f"DEBUG: Data inserted successfully into table: {table_id}")
"""

    def post(self, request, *args, **kwargs):
        user_id = request.data.get("user_id")
        uploaded_files = request.FILES.getlist("files")

        print(f"DEBUG: Received upload request for user: {user_id}, files: {len(uploaded_files)}")

        # 🔍 Validate user authentication
        user = None#bq_auth_handler.get_user_from_id(user_id)
        if not user:
            print(f"DEBUG: Invalid user ID: {user_id}")
            return Response({"error": "Invalid user ID"}, status=403)


        # 🔍 Validate number of files
        if len(uploaded_files) == 0:
            print("DEBUG: No files uploaded")
            return Response({"error": "No files uploaded"}, status=400)
        if len(uploaded_files) > MAX_FILES:
            print(f"DEBUG: Too many files uploaded: {len(uploaded_files)}")
            return Response({"error": f"Too many files uploaded. Max {MAX_FILES} allowed."}, status=400)

        processed_files = []

        for file in uploaded_files:
            print(f"DEBUG: Processing file: {file.name}")
            file_extension = file.name.split(".")[-1].lower()
            file_size_mb = file.size / (1024 * 1024)  # Convert bytes to MB

            # 🔍 Validate file type
            if file_extension not in ALLOWED_EXTENSIONS:
                print(f"DEBUG: Unsupported file type: {file_extension}")
                return Response({"error": f"Unsupported file type: {file_extension}"}, status=400)

            # 🔍 Validate file size
            if file_size_mb > MAX_FILE_SIZE_MB:
                print(f"DEBUG: File {file.name} exceeds size limit: {file_size_mb}MB")
                return Response({"error": f"File {file.name} exceeds {MAX_FILE_SIZE_MB}MB limit"}, status=400)

            ######Uplad file

            local_dir = "./data/files/"
            os.makedirs(local_dir, exist_ok=True)
            file_path = local_dir+file.name
            with open(file_path, "wb") as f:
                f.write(file.read())
            bucket_path=f"users/{user_id}/files"
            #bucket.upload_file(file_path, remote_path=bucket_path)
            os.remove(file_path)

            try:
                extracted_text = process_document_sample(
                    project_id=GCP_ID,
                    location="us",
                    processor_display_name=user_id,
                    file_prefix=bucket_path,  # Process directly from GCS
                    mime_type="application/pdf",
                )

                print(f"DEBUG: Document AI processing successful for: {file.name}")
            except Exception as e:
                print(f"DEBUG: Error processing {file.name}: {str(e)}")
                return Response({"error": f"Error processing {file.name}: {str(e)}"}, status=500)

            u_d=[]
            table_id = user_id + "_files"
            for i, chunk in enumerate(chunk_text(extracted_text, chunk_size=350)):
                chunk_id = f"{file.name}_{i}_{date.today()}"
                embedding_np = None
                embedding_list = [float(x) for x in embedding_np.tolist()]
                u_d.append({
                    "id": chunk_id,
                    "file": file.name,
                    "text": chunk,
                    "embedding": embedding_list
                })
                print(f"DEBUG: Chunk created: {chunk_id}")

            self.handle_file_insertion_bq(u_d, table_id=table_id)
            print(f"DEBUG: All chunks inserted for file: {file.name}")

            response = requests.post(REQUEST_URL+"/work/emb/", {"table_id":table_id})
            print("Embedding response", response)

        print("DEBUG: All files processed successfully")
        return Response({"success": True, "processed_files": processed_files}, status=200)


def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
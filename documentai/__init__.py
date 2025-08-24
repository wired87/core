# pip install ipython google-cloud-documentai tabulate

import dotenv
dotenv.load_dotenv()

import os
import google.cloud.documentai_v1 as docai



API_LOCATION="eu"
os.environ["API_LOCATION"]=API_LOCATION


def get_client() -> docai.DocumentProcessorServiceClient:
    client_options = {"api_endpoint": f"{API_LOCATION}-documentai.googleapis.com"}
    return docai.DocumentProcessorServiceClient(client_options=client_options)

def get_parent(client: docai.DocumentProcessorServiceClient) -> str:
    return client.common_location_path(
        os.environ.get("GCP_PROJECT_ID"),
        API_LOCATION
    )

ENDPOING="https://us-documentai.googleapis.com/v1/projects/1004568990634/locations/us/processors/97d3c99a5d0bb913:process"


DOCAI_CLIENT = get_client()
DOCAI_PARENT = get_parent(DOCAI_CLIENT)
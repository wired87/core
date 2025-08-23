# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# [START documentai_process_document]
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.cloud import documentai, bigquery, storage  # type: ignore

# TODO(_b): Uncomment these variables before running the sample.
# project_id = "YOUR_PROJECT_ID"
# location = "YOUR_PROCESSOR_LOCATION" # Format is "us" or "eu"
# processor_id = "YOUR_PROCESSOR_ID" # Create processor before running sample
# file_path = "/path/to/local/pdf"
# mime_type = "application/pdf" # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
# field_mask = "text,entities,pages.pageNumber"  # Optional. The fields to return in the Document object.
# processor_version_id = "YOUR_PROCESSOR_VERSION_ID" # Optional. Processor version to use

def get_processor_id_by_display_name(client, project_id, location, display_name):
    """Retrieves the processor ID based on the display name."""
    parent = f"projects/{project_id}/locations/{location}"
    for processor in client.list_processors(parent=parent):
        if processor.display_name == display_name:
            return processor.name.split("/")[-1]  # Extract processor ID
    return None  # Processor not found

def process_document_sample(
    project_id: str,
    location: str,
    processor_display_name: str,  # Use display name instead of ID
    file_prefix: str,  # Cloud Storage URI (gs://bucket/path/to/file)
    mime_type: str,
    field_mask: Optional[str] = None,
    processor_version_id: Optional[str] = None,
) -> Optional[str]:
    # You must set the `api_endpoint` if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # Get processor ID from display name
    processor_id = get_processor_id_by_display_name(client, project_id, location, processor_display_name)
    if not processor_id:
        print(f"Processor with display name '{processor_display_name}' not found.")
        return None

    if processor_version_id:
        name = client.processor_version_path(
            project_id, location, processor_id, processor_version_id
        )
    else:
        name = client.processor_path(project_id, location, processor_id)

    # Load file from Cloud Storage
    storage_client = storage.Client(project=project_id)

    bucket = storage_client.bucket("bestbrain")
    blob = bucket.blob(file_prefix)
    content = blob.download_as_bytes()

    # Load binary data
    raw_document = documentai.RawDocument(content=content, mime_type=mime_type, display_name=blob.name)

    # For more information: https://cloud.google.com/document-ai/docs/reference/rest/v1/ProcessOptions
    process_options = documentai.ProcessOptions(
        individual_page_selector=documentai.ProcessOptions.IndividualPageSelector(
            pages=[1]
        )
    )

    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document,
        field_mask=field_mask,
        process_options=process_options,
    )

    result = client.process_document(request=request)
    document = result.document

    print("The document:", document)
    print(document.text)
    return document.text


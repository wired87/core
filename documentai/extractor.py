from google.cloud import documentai_v1 as documentai

from docai import DOCAI_CLIENT


def get_content(attachments):
    extracted_data=[]
    for attachment in attachments:
        file_content = attachment.read()
        mime_type = attachment.content_type or "application/octet-stream"
        raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)

        try:
            proc_request = documentai.ProcessRequest(
                name="projects/1004568990634/locations/eu/processors/1838757af9bea563",
                raw_document=raw_document
            )
            response = DOCAI_CLIENT.process_document(request=proc_request)
            # print("response", response)
            if response.document.entities:
                # print("response.document.entities", response.document.entities)
                print("entities found:", len(response.document.entities))
                extracted_data.append(response.document)

        except Exception as e:
            print(f"Verarbeitung von '{attachment.name}' fehlgeschlagen: {e}")
            extracted_data.append(
                {"Originaldateiname": attachment.name, "Fehler": "Verarbeitung fehlgeschlagen"}
            )
    print("All attachments extracted")
    return extracted_data


def extract_keys(whole_content):

    keys = []
    for content in whole_content:
        for e in content.entities:
            keys.append(
                e.type_
            )

    return keys

def convert_rows(whole_content, all_keys):
    rows = []

    for content in whole_content:
        row = []
        for key in all_keys:
            for e in content.entities:
                if e.type_ == key:
                    row.append(e.mention_text)
        rows.append(row)
    return rows

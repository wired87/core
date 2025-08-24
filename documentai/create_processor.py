#Supports JPEG, JPG, PNG, BMP, PDF, TIFF, TIF, GIF (15 pages, 20MB max)

from google.cloud import documentai
from _google import GCP_ID
from documentai import DOCAI_CLIENT


def get_create_processor(display_name):
    existing_processor = check_processor_exists(display_name=display_name)
    if isinstance(existing_processor, bool) and existing_processor is False:
        # create processor
        return create_processor(
            processor_display_name=display_name
        )
    if isinstance(existing_processor, dict):
        return existing_processor




def create_processor(
        location: str = "eu",
        processor_display_name: str = "Mein-Prozessor",
        processor_type: str = "OCR_PROCESSOR",
):
    """
    Erstellt einen neuen Document AI Prozessor mit allen verfügbaren Parametern.
    """

    # Erstellt ein Processor-Objekt mit allen Parametern
    processor_config = documentai.Processor(
        display_name=processor_display_name,
        type_=processor_type,
    )

    parent = DOCAI_CLIENT.common_location_path(GCP_ID, location)

    try:
        processor = DOCAI_CLIENT.create_processor(
            parent=parent,
            processor=processor_config,
        )

        print(f"✅ Prozessor '{processor.display_name}' ({processor.type_}) wurde erfolgreich erstellt: {processor.name}")
        return dict(
            name=processor.name,
            type=processor.type_,
            endpoint=processor.process_endpoint
        )

    except Exception as e:
        print(f"❌ Fehler beim Erstellen des Prozessors: {e}")
        raise

def check_processor_exists(display_name, location="eu"):
    # Der vollständige Ressourcenname des Standorts
    parent = DOCAI_CLIENT.common_location_path(GCP_ID, location)

    # Ruft alle Prozessoren am angegebenen Standort ab
    for processor in DOCAI_CLIENT.list_processors(parent=parent):
        if processor.display_name == display_name:
            print(f"Prozessor '{display_name}' existiert bereits.")
            return dict(
                name=processor.name,
                type=processor.type_,
                endpoint=processor.process_endpoint
            )
    print(f"Prozessor '{display_name}' existiert nicht.")
    return False
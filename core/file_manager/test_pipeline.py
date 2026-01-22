import base64
import os
import pprint
from core.file_manager.file_lib import file_manager

def test_file_manager_pipeline():
    print("Testing File Manager Pipeline...")
    
    # Path to local image
    image_path = os.path.join(os.path.dirname(__file__), "image.png")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Read and encode image
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    # Construct data payload
    # Mocking standard frontend input with data URI
    files = [f"data:image/png;base64,{encoded_string}"]
    
    data = {
        "id": "test_module_id_123",
        "name": "Test Module from Image",
        "description": "A module extracted from image.png for testing.",
        "files": files
    }
    
    user_id = "test_user"
    
    # Run pipeline with testing=True
    result = file_manager.process_and_upload_file_config(
        user_id=user_id,
        data=data,
        testing=True
    )
    
    print("\n\n--- Test Results ---")
    pprint.pprint(result)

if __name__ == "__main__":
    test_file_manager_pipeline()


import os
import sys
import json
import base64
import logging

# Ensure project root is in sys.path
sys.path.append(os.getcwd())

from core.module_manager.module_etractor.extractor import RawModuleExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_extractor():
    print("Starting RawModuleExtractor Test...")

    # Configuration
    USER_ID = "72b74d5214564004a3a86f441a4a112f"
    MODULE_ID = "gravity_mod_ex" 
    
    # Use the local copy we just verified
    USER_PDF_PATH = rf"C:\Users\bestb\PycharmProjects\BestBrain\core\module_manager\module_etractor\2601.00586v1.pdf"
    
    if os.path.exists(USER_PDF_PATH):
        PDF_PATH = USER_PDF_PATH
        print(f"Using local PDF: {PDF_PATH}")
    else:
        raise Exception(f"PDF file not found at {USER_PDF_PATH}")

    OUTPUT_FILE = os.path.join(os.getcwd(), "core", "module_manager", "module_etractor", "test_out.json")

    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return

    # Read PDF
    try:
        with open(PDF_PATH, "rb") as f:
            pdf_bytes = f.read()
            print(f"Read PDF bytes: {len(pdf_bytes)}")
            # Encode to base64
            pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
            
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    # Initialize Extractor
    print(f"Initializing Extractor for User: {USER_ID}, Module: {MODULE_ID}")
    extractor = RawModuleExtractor()

    # Execute Pipeline
    print("Executing pipeline (Gemini requests + BQ Ops)...")
    try:
        result = extractor.process(files=[pdf_b64], mid=MODULE_ID)
        
        print("Pipeline execution completed.")
        print("Result keys:", result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return

    print("Write Output")
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Results written to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    test_extractor()

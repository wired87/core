
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _gmail.gmail_processor import GmailProcessor

def test_workflow():
    print("Testing GmailProcessor workflow...")
    
    # Path to credentials - adjusting based on project structure
    # Assuming running from root or _gmail
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    creds_path = os.path.join(base_path, "auth", "credentials.json")
    token_path = os.path.join(base_path, "auth", "token.pickle")
    
    print(f"Using credentials path: {creds_path}")

    processor = GmailProcessor()
    
    # Test content generation methods
    success_content = processor.get_success_content("test_model", "deployed")
    print("Success content generated successfully.")
    
    failed_content = processor.get_crawl_failed_content()
    print("Failed content generated successfully.")
    
    email_html = processor.get_email("Heading", "Body Heading", "Body Text")
    print("Email HTML generated successfully.")

    # Test sending email (will skip if service not initialized)
    if processor.service:
        print("Service initialized, attempting to send email...")
        recipient = "bestfbneu@gmail.com"
        result = processor.gmail_send_message(recipient, email_html, "Test Subject")
        print(f"Send email result: {result}")
    else:
        print("Service not initialized (likely no valid credentials/token found or interactive auth required), skipping send.")

    # Test other methods logic
    try:
        processor.webhook_missing_fields()
        print("webhook_missing_fields executed.")
    except Exception as e:
        print(f"webhook_missing_fields failed: {e}")

if __name__ == "__main__":
    test_workflow()

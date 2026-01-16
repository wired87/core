from _gmail.gmail_processor import GmailProcessor

processor = GmailProcessor()

RELAY_GMAIL = [
    {
        "case_id": "SEND_BOT_FAILED",
        "action": processor.send_bot_failed,
        "input_map": {
            "bot_model_name": "bot_model_name",
            "user_email": "user_email"
        }
    },
    {
        "case_id": "DEV_EMAIL_ACTOR_FAILED",
        "action": processor.dev_email_actor_failed,
        "input_map": {
            "bot_model_name": "bot_model_name"
        }
    },
    {
        "case_id": "SEND_CRAWL_FAILED",
        "action": lambda user_email: processor.gmail_send_message(
            user_email,
            processor.get_crawl_failed_content(),
            "Crawl Failed Notification"
        ),
        "input_map": {
            "user_email": "user_email"
        }
    },
     {
        "case_id": "SEND_MODEL_DEPLOYMENT_FAILED",
        "action": lambda user_email, model_name: processor.gmail_send_message(
            user_email,
            processor.get_model_deployment_failed_content(model_name),
            "Deployment Failed Notification"
        ),
        "input_map": {
            "user_email": "user_email",
            "model_name": "model_name"
        }
    },
    {
        "case_id": "SEND_SUCCESS_CONTENT",
        "action": lambda user_email, model_name, status: processor.gmail_send_message(
            user_email,
            processor.get_success_content(model_name, status),
            "Deployment Successful Notification"
        ),
        "input_map": {
            "user_email": "user_email",
             "model_name": "model_name",
             "status": "status"
        }
    }
]

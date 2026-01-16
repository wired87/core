import base64
import os
import pickle
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

from auth.load_sa_creds import load_service_account_credentials

SCOPES = ['https://www.googleapis.com/auth/gmail.send']


class GmailProcessor:

    def __init__(
            self,
    ):
        # Load environment variables
        load_dotenv()
        self.dev_email = os.getenv("DEV_EMAIL", "info@botworld.cloud")

        # Prefer env vars over defaults, but allow override if args passed strictly (though logic here favors env if set, typical for 12 factor apps or use args as fallback)
        # Actually, let's allow args to be the defaults if Env not set.
        self.credentials_path = os.getenv("GC", None)
        self.scopes = ["https://www.googleapis.com/auth/gmail.send"]

        self.service = self._create_service()


    def _create_service(self):
        print("CREATE SERVICE...")
        creds:Credentials = load_service_account_credentials(
            self.credentials_path,
            self.scopes,
        )
        if not creds:
            print("Failed to obtain credentials.")
            return None
        creds = creds.with_subject(self.dev_email)
        try:
            service = build(
                serviceName='gmail',
                version='v1',
                credentials=creds,
            )
            print("SERVICE CREATED:", service)
            return service
        except Exception as e:
            print("COULDNT CREATE A SERVICE CAUSE ERROR:", e)
            return None

    def create_and_send_email_content(self, email: str, html_content: str, subject: str) -> bool:
        if not self.service:
            print("Gmail service is not initialized.")
            return False

        try:
            message = MIMEMultipart('alternative')
            print("EMAIL MESSAGE INSTANCE CREATED...")

            message['To'] = email
            message['From'] = "Chef Etage"  # Could also be parameterized
            message['Subject'] = subject
            print("EMAIL DATA CREATED...")

            # Set the HTML content
            part = MIMEText(html_content, 'html')
            message.attach(part)

            # Encode the message in base64url
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            print("MESSAGE ENCODED...")

            # Send the message
            send_message = (
                self.service.users()
                .messages()
                .send(userId="me", body={"raw": raw_message})
                .execute()
            )
            print(f'MESSAGE RESPONSE: {send_message}')
            return True

        except HttpError as e:
            print(f"AN HTTP REQUEST ERROR HAS OCCURRED: {e}")
            return False
        except Exception as e:
            print("UNEXPECTED ERROR OCCURRED:", e)
            return False

    def gmail_send_message(self, email: str, content: str, subject: str) -> bool:
        print("SEND GMAIL STARTED...")
        if not self.service:
            # Try to re-init service if missing?
            self.service = self._create_service()
            if not self.service:
                return False

        return self.create_and_send_email_content(email, content, subject)

    def dev_email_actor_failed(self, bot_model_name):
        self.gmail_send_message(
            self.dev_email,
            self.get_email(
                heading="Deployment failed",
                body_heading=f"Bot Name {bot_model_name}",
                body_text=f"Time: {datetime.now()} \n Context: IN APify Webhooks Failed event was given."
            ),
            f"Bot deployment has been failed"
        )

    def send_bot_failed(self, bot_model_name, user_email):
        self.gmail_send_message(
            user_email,
            self.get_email(
                heading="Unfortunately the deployment of your bot has been failed.",
                body_heading="Sometimes we have a stormy weather in the cloud which results sometimes in issues."
                             "We're sorry about that and have informed our Dev-Team to fix it asap"
                             "Your Bot limit has been ",
                body_text=""
            ),
            f"Bot deployment has been failed"
        )
        self.dev_email_actor_failed(
            bot_model_name=bot_model_name
        )

    def get_success_content(self, model_name, status):
        return f"""
            The Website data has been successfully collected.\n\n

            Your Model with ID: {model_name} instance has been created and is currently under status: {status}. \n\n

            You can follow the status of deployment in your Dashboard under the following url:\n
            https://DASHBOARD-URL

            We will contact you about any changes.

        """

    def get_crawl_failed_content(self):
        return f"""
            The scrapping of the data, unfortunately has failed.\n\n

            The Dev - Team is informed and will check it asap.

            We will contact you about any changes.

            Best regards, \n
            Your BotWorld - Team
        """

    def dev_team_crawl_error_message(self):
        return f"""
            The data scrapping of the data for model, unfortunately has failed.\n\n

            DATA:

        """

    def get_model_deployment_failed_content(self, model_id: str):
        return f"""
            The deployment of the model with id: {model_id}, unfortunately has failed.\n\n

            The Dev - Team is informed about the issue and will fix it asap.

            We will contact you about any changes.

            Best regards, \n
            Your BotWorld - Team
        """

    def dev_team_deployment_error_message(self, model_name, time):
        return f"""
            The deployment of the model with ID {model_name}, unfortunately has failed.\n\n

            DATA:
            Time: {time}
        """

    def dev_team_worker_connect_error(self, error, time):
        return f"""
            Could not connect to the worker as Time: {time} cause eth following error while handling the deployment process {error}: 

            DATA:

        """

    def webhook_missing_fields(self):
        # using datetime.now() instead of timezone.now() to avoid django dependency
        return f"""
        Webhook is missing fields. The process couldn't be handled at time {datetime.now()}
        """

    def _get_email_footer(self):
        return f"""
            <tr>
                <td class="footer">
                    <p>&copy; {datetime.now().year} ClusterExpress. High-Performance Equation Infrastructure.</p>
                    <p>
                        <a href="#">Security</a> | 
                        <a href="#">API Docs</a> | 
                        <a href="#">GCP Status</a>
                    </p>
                    <p style="margin-top: 15px; opacity: 0.6;">
                        This is an automated notification from your production environment.
                    </p>
                </td>
            </tr>
        """
        
    def _get_button(self, text, url):
        return f"""
            <div class="button-container">
                <a href="{url}" target="_blank" class="btn">{text}</a>
            </div>
        """
        
    def _get_module_card(self, title, status, status_color="#10b981"):
        return f"""
            <div class="module-card">
                <span class="module-title">{title}</span>
                <p style="margin: 0; font-size: 14px;">Status: <span style="color: {status_color};">● {status}</span></p>
            </div>
        """

    def get_email(
            self,
            heading: str,
            body_heading: str,
            body_text: str
    ) -> str:
        return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f4f4f4;
                    }}
                    .email-container {{
                        max-width: 600px;
                        margin: 20px auto;
                        background-color: #ffffff;
                        border-radius: 8px;
                        overflow: hidden;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }}
                    .header {{
                        background-color: #09175F;
                        color: #ffffff;
                        padding: 20px;
                        text-align: center;
                    }}
                    .content {{
                        padding: 20px;
                        line-height: 1.6;
                        color: #333333;
                    }}
                    .content h2 {{
                        color: #09175F;
                    }}
                    .footer {{
                        background-color: #f0f0f0;
                        color: #666666;
                        padding: 20px;
                        text-align: center;
                        font-size: 12px;
                    }}
                    .footer a {{
                        color: #09175F;
                        text-decoration: none;
                        margin: 0 5px;
                    }}
                    .button-container {{
                        text-align: center;
                        margin: 20px 0;
                    }}
                    .btn {{
                        display: inline-block;
                        background-color: #09175F;
                        color: #ffffff;
                        padding: 10px 20px;
                        border-radius: 5px;
                        text-decoration: none;
                        font-weight: bold;
                    }}
                    .module-card {{
                        background-color: #f9f9f9;
                        border: 1px solid #e0e0e0;
                        border-radius: 5px;
                        padding: 15px;
                        margin-bottom: 10px;
                    }}
                    .module-title {{
                        font-weight: bold;
                        color: #09175F;
                        margin-bottom: 5px;
                        display: block;
                    }}
                </style>
            </head>
            <body>
                <div class="email-container">
                    <div class="header">
                        <h1>{heading}</h1>
                    </div>
                    <div class="content">
                        <h2>{body_heading}</h2>
                        <p>{body_text}</p>
                    </div>
                    {self._get_email_footer()}
                </div>
            </body>
            </html>
        """

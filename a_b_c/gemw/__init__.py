import os

from google import genai
import dotenv as dotenv
dotenv.load_dotenv()

_gemini_key = os.environ.get("GEMINI_API_KEY")
GAIC = genai.Client(api_key=_gemini_key) if _gemini_key else None

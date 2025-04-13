import os
import logging

try:
    google_genai_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    google_genai_model_smaller = os.getenv("google_genai_model_smaller", "gemini-1.5-flash")
    google_genai_model_better = os.getenv("google_genai_model_smaller", "gemini-2.0-flash")
except:
    logging.error(f"Please configure environment variables properly!")

import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
headers = {
    "anthropic-version": "2023-06-01",
    "x-api-key": api_key
}

response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
# print(response.status_code)
# print(response.text)
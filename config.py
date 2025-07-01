# config.py
import os
from dotenv import load_dotenv
load_dotenv()  # loads from .env if present
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
CLAUDE_API_KEY  = os.environ.get("CLAUDE_API_KEY")
BRAVE_API_KEY   = os.environ.get("BRAVE_API_KEY")

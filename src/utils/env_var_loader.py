""" Module to load Env Variables. """
import os

from dotenv import load_dotenv

load_dotenv()

# Import Env Variables
GEMINI_APY_KEY = os.environ.get("GOOGLE_API_KEY")

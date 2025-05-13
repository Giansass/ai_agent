""" Module to load Env Variables. """
import os

from dotenv import load_dotenv

load_dotenv()

# Import Env Variables
GEMINI_APY_KEY = os.environ.get("GOOGLE_API_KEY")
LLM_MODEL_PROVIDER = os.environ.get("LLM_MODEL_PROVIDER")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME")
AGENT_TO_BE_EXECUTED = os.environ.get("AGENT_TO_BE_EXECUTED")
PHOENIX_COLLECTOR_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")

""" Module to load Env Variables. """
import os

from dotenv import load_dotenv

load_dotenv()

# Import Env Variables
GEMINI_APY_KEY = os.environ.get("GOOGLE_API_KEY")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME",
                                "models/gemini-2.0-flash-lite")
AGENT_TO_BE_EXECUTED = os.environ.get("AGENT_TO_BE_EXECUTED")
PHOENIX_COLLECTOR_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
WEB_SEARCH_ENGINE = os.environ.get("WEB_SEARCH_ENGINE", "Bing")

# Chroma Env Variables
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME",
                                        "web_search_collection")
CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY",
                                          "chroma_db/web_search_collection")


# Import Bing Search Env Variables
BING_SUBSCRIPTION_KEY = os.environ.get("BING_SUBSCRIPTION_KEY", "No key provided")
BING_SEARCH_URL = os.environ.get("BING_SEARCH_URL",
                                 "https://api.bing.microsoft.com/v7.0/search")
BING_HEADERS = {"Ocp-Apim-Subscription-Key": BING_SUBSCRIPTION_KEY}
BING_SIZE_PAGE = int(os.environ.get("BING_SIZE_PAGE", 10))
BING_N_PAGES = int(os.environ.get("BING_N_PAGES", 1))
BING_FROM_DATE = os.environ.get("BING_FROM_DATE", "2022-01-01")
BING_TO_DATE = os.environ.get("BING_TO_DATE", "2025-06-20")

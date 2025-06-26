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
WEB_SEARCH_ENGINE = os.environ.get("WEB_SEARCH_ENGINE", "DuckDuckGo")
WORKFLOW_TIMEOUT = int(os.environ.get("WORKFLOW_TIMEOUT", 30))
DUCKDUCKGO_CERTIFICATE_PATH = os.environ.get("DUCKDUCKGO_CERTIFICATE_PATH")

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

# Import Token Splitter Env Variables
TOKEN_SPLITTER_CHUNK_SIZE = int(os.environ.get("TOKEN_SPLITTER_CHUNK_SIZE", 1024))
TOKEN_SPLITTER_CHUNK_OVERLAP = int(os.environ.get("TOKEN_SPLITTER_CHUNK_OVERLAP", 100))
TOKEN_SPLITTER_MODEL_NAME = os.environ.get("TOKEN_SPLITTER_MODEL_NAME", "o200k_base")

EMBEDDING_TOKEN_DOLLAR_PRICE_PER_1M = float(
    os.environ.get("EMBEDDING_TOKEN_DOLLAR_PRICE_PER_1M", 20)
)

PROMPT_TOKEN_DOLLAR_PRICE_PER_1M = float(
    os.environ.get("TOKEN_DOLLAR_PRICE_PER_1M", 0.075)
)
COMPLETITION_TOKEN_DOLLAR_PRICE_PER_1M = float(
    os.environ.get("TOKEN_DOLLAR_PRICE_PER_1M", 0.3)
)

# Import DuckDuckGo Env Variables
DUCKDUCKGO_CERTIFICATE_PATH = os.environ.get("DUCKDUCKGO_CERTIFICATE_PATH",
                                             "/Users/a473589/.certificates/global_certificates.pem")
DUCKDUCKGO_MAX_RESULTS = int(os.environ.get("DUCKDUCKGO_MAX_RESULTS", 5))

#  Import top K companies and MG
TOP_K_URLS = int(os.environ.get("TOP_K_URLS", 10))
TOP_K_MG = int(os.environ.get("TOP_K_MG", 10))

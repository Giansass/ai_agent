"""aaa"""

from src.utils.env_var_loader import GEMINI_APY_KEY, LLM_MODEL_NAME, LLM_MODEL_PROVIDER
from src.utils.llm_loader import LLMLoader

# Large Language Model initialization
llm_Loader = LLMLoader(
    model_provider=LLM_MODEL_PROVIDER,
    model_name=LLM_MODEL_NAME,
    api_key=GEMINI_APY_KEY,
)

llm = llm_Loader.llm_initializer()

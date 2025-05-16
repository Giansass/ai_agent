"""aaa"""

from src.utils.env_var_loader import GEMINI_APY_KEY, LLM_MODEL_NAME
from src.utils.llm_loader import LLMLoader

# Large Language Model initialization
llm_Loader = LLMLoader(
    model_name=LLM_MODEL_NAME,
    api_key=GEMINI_APY_KEY,
)

llm_text_generation = llm_Loader.llm_initializer(model_type="text_generation")
llm_text_embedding = llm_Loader.llm_initializer(model_type="embedding")

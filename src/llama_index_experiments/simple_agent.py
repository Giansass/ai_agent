"""src base module."""
from llama_index.core.agent.workflow import FunctionAgent

from src.utils.env_var_loader import GEMINI_APY_KEY, LLM_MODEL_NAME, LLM_MODEL_PROVIDER
from src.utils.llm_loader import LLMLoader

# Large Language Model initialization
llm_Loader = LLMLoader(
    model_name=LLM_MODEL_NAME,
    api_key=GEMINI_APY_KEY,
)

# Load llm
if LLM_MODEL_PROVIDER == "google_genai":
    llm = llm_Loader.gemini_model_loader()
else:
    raise ValueError(f"Model provider {LLM_MODEL_PROVIDER} not supported")


# Tools definitions
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


# Workflow definition
workflow = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools.",
)


async def main():
    """Main function"""
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

"""src base module."""
from llama_index.core.agent.workflow import FunctionAgent

from src.llama_index_experiments.llm_load import llm


# Tools definitions
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


SYSTEM_PROMPT = """
You are an agent that can perform basic mathematical operations using tools.
"""

# Workflow definition
workflow = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt=''.join(SYSTEM_PROMPT.splitlines()),
)


async def simple_agent_execution():
    """Main function"""
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(simple_agent_execution())

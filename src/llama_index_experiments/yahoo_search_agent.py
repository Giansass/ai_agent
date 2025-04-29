"""src base module."""
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

from src.llama_index_experiments.llm_load import llm

yahoo_tool_spec = YahooFinanceToolSpec().to_tool_list()

# FunctionAgent definition
workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    llm=llm,
    tools=yahoo_tool_spec,
    system_prompt="You are a helpful assistant.",
)


async def yahoo_agent_execution():
    """Main function"""
    response = await workflow.run(user_msg="What is the current price of NVIDIA stock?")
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(yahoo_agent_execution())

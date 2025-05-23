"""src base module."""
import os
from dotenv import load_dotenv

load_dotenv()

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

gemini_api_key = os.environ.get('GOOGLE_API_KEY')

llm = GoogleGenAI(
    model="models/gemini-2.0-flash",
    api_key=gemini_api_key,
)

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

workflow = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools.",
)

async def main():
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
"""Module used to load llm models"""
from typing import Optional

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI


class LLMLoader:
    """LLM Loader class"""

    def __init__(self, model_name: Optional[str], api_key: Optional[str]):
        """LLM Loader constructor"""
        self.model_name = model_name
        self.api_key = api_key

    def gemini_model_loader(self) -> GoogleGenAI:
        """Load Gemini LLM"""
        return GoogleGenAI(
            model=self.model_name,
            api_key=self.api_key,
        )

    def open_ai_model_loader(self) -> OpenAI:
        """Load Open AI LLM"""
        return OpenAI(
            model=self.model_name,
            api_key=self.api_key,
        )

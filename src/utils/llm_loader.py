"""Module used to load llm models"""
from typing import Optional, Union

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI


class LLMLoader:
    """LLM Loader class"""

    def __init__(
        self,
        model_provider: Optional[str],
        model_name: Optional[str],
        api_key: Optional[str],
    ):
        """LLM Loader constructor"""
        self.model_provider = model_provider
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

    def llm_initializer(self) -> Union[GoogleGenAI, OpenAI]:
        """Main method"""
        if self.model_provider == "google_genai":
            return self.gemini_model_loader()
        if self.model_provider == "openai":
            return self.open_ai_model_loader()

        raise ValueError(f"Model provider {self.model_provider} not supported")

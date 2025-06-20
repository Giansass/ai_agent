"""Module used to load llm models"""
from typing import Optional, Union

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI


class LLMLoader:
    """LLM Loader class"""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str],
    ):
        """LLM Loader constructor"""
        self.model_name = model_name
        self.api_key = api_key

    def gemini_model_loader(self) -> GoogleGenAI:
        """Load Gemini LLM"""
        return GoogleGenAI(
            model=self.model_name,
            api_key=self.api_key,
        )

    def gemini_embedding_model_loader(self) -> GoogleGenAIEmbedding:
        """Load Open AI LLM"""
        return GoogleGenAIEmbedding(
            model=self.model_name,
            api_key=self.api_key,
        )

    def llm_initializer(
            self,
            model_type: str) -> Union[GoogleGenAI, GoogleGenAIEmbedding]:
        """Main method"""
        if model_type == "text_generation":
            return self.gemini_model_loader()
        if model_type == "embedding":
            return self.gemini_embedding_model_loader()

        raise ValueError(f"Model provider {model_type} not supported")

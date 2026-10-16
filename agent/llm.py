import os
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())


class LLMModelProvider(ABC):
    @abstractmethod
    def create_model(self):
        pass


class OpenAIModelProvider(LLMModelProvider):
    def create_model(
        self,
        model: str = os.getenv("LLM_MODEL"),
        temperature: float = float(os.getenv("LLM_TEMPERATURE")),
        max_tokens: int = int(os.getenv("LLM_MAX_TOKENS")),
    ) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

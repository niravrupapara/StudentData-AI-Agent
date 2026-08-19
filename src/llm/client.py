# src/llm/client.py

from langchain_mistralai import ChatMistralAI

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


def create_llm() -> ChatMistralAI:
    """
    Create and return the configured Mistral LLM.
    """

    if not settings.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not configured.")

    logger.info(
        "Initializing LLM: %s",
        settings.LLM_MODEL,
    )

    llm = ChatMistralAI(
        model=settings.LLM_MODEL,
        api_key=settings.MISTRAL_API_KEY,
        temperature=0,
    )

    logger.info("LLM initialized successfully.")

    return llm


llm = create_llm()
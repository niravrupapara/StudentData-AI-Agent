from langchain_mistralai import ChatMistralAI

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_llm() -> ChatMistralAI:
    """Initialize and return the configured Mistral LLM."""

    logger.info(
        "Initializing LLM: %s",
        settings.LLM_MODEL,
    )

    if not settings.MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY is missing.")
        raise ValueError(
            "MISTRAL_API_KEY is not configured. "
            "Check your .env file."
        )

    try:
        llm = ChatMistralAI(
            model=settings.LLM_MODEL,
            temperature=0,
            api_key=settings.MISTRAL_API_KEY,
        )

        logger.info("LLM initialized successfully.")

        return llm

    except Exception:
        logger.exception("Failed to initialize LLM.")
        raise

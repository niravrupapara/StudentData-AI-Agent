# config/settings.py

import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Application configuration."""

    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")

    LLM_MODEL: str = os.getenv(
        "LLM_MODEL",
        "mistral-small-latest",
    )

    EMBEDDING_MODEL: str = "./models/all-MiniLM-L6-v2"

    TOP_K: int = int(os.getenv("TOP_K", "5"))

    LOG_LEVEL: str = os.getenv(
        "LOG_LEVEL",
        "INFO",
    )


settings = Settings()
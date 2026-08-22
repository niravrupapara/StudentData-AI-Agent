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

    LOG_LEVEL: str = os.getenv(
        "LOG_LEVEL",
        "INFO",
    )


settings = Settings()
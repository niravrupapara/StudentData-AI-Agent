# config/settings.py

from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

with open(BASE_DIR / "config" / "config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


class Settings:
    """Application configuration."""

    APP_NAME = config["app"]["name"]

    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

    LLM_PROVIDER = config["llm"]["provider"]
    LLM_MODEL = config["llm"]["model"]
    LLM_TEMPERATURE = config["llm"]["temperature"]

    MAX_RETRIES = config["agent"]["max_retries"]

    PDF_CHUNK_SIZE = config["pdf"]["chunk_size"]
    PDF_CHUNK_OVERLAP = config["pdf"]["chunk_overlap"]
    PDF_RETRIEVAL_K = config["pdf"]["retrieval_k"]


settings = Settings()

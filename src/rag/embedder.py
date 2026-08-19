# src/rag/embedder.py

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and return the configured embedding model.
    """

    logger.info(
        "Loading embedding model: %s",
        settings.EMBEDDING_MODEL,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    logger.info("Embedding model loaded successfully.")

    return embeddings


# Load the embedding model once and reuse it.
embeddings = create_embedding_model()
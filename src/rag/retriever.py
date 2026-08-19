# src/rag/retriever.py

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


def retrieve_documents(
    vector_store: FAISS,
    query: str,
    top_k: int | None = None,
) -> list[Document]:
    """
    Retrieve the most relevant documents for a query.

    Args:
        vector_store: FAISS vector store.
        query: User's natural-language query.
        top_k: Number of documents to retrieve.

    Returns:
        List of relevant documents.
    """

    if not query.strip():
        logger.warning("Empty retrieval query received.")
        return []

    k = top_k or settings.TOP_K

    logger.info(
        "Starting RAG retrieval | top_k=%d | query=%s",
        k,
        query,
    )

    documents = vector_store.similarity_search(
        query,
        k=k,
    )

    logger.info(
        "RAG retrieval completed | documents=%d",
        len(documents),
    )

    return documents
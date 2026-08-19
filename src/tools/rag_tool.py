# src/tools/rag_tool.py

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import settings
from src.rag.retriever import retrieve_documents
from src.utils.logger import get_logger


logger = get_logger(__name__)


def execute_rag_query(
    vector_store: FAISS,
    query: str,
    top_k: int | None = None,
) -> list[Document]:
    """
    Retrieve relevant documents for a user query.

    Args:
        vector_store: FAISS vector store.
        query: User's natural-language query.
        top_k: Number of documents to retrieve.

    Returns:
        List of relevant documents.
    """

    if vector_store is None:
        raise ValueError("Vector store is not initialized.")

    if not query or not query.strip():
        logger.warning("Empty RAG query received.")
        return []

    k = top_k or settings.TOP_K

    logger.info(
        "Executing RAG tool | top_k=%d",
        k,
    )

    try:
        documents = retrieve_documents(
            vector_store=vector_store,
            query=query,
            top_k=k,
        )

        logger.info(
            "RAG tool completed | documents=%d",
            len(documents),
        )

        return documents

    except Exception:
        logger.exception("RAG tool execution failed.")
        raise
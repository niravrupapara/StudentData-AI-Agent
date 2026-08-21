# src/tools/rag_tool.py

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool

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
    """

    if vector_store is None:
        raise ValueError("Vector store is not initialized.")

    if not query or not query.strip():
        logger.warning("Empty RAG query received.")
        return []

    k = top_k or settings.TOP_K

    logger.info(
        "Executing RAG query | top_k=%d",
        k,
    )

    try:
        documents = retrieve_documents(
            vector_store=vector_store,
            query=query,
            top_k=k,
        )

        logger.info(
            "RAG query completed | documents=%d",
            len(documents),
        )

        return documents

    except Exception:
        logger.exception("RAG query execution failed.")
        raise


def create_rag_tool(vector_store: FAISS):
    """
    Create a RAG tool bound to the application's vector store.
    """

    @tool
    def rag_tool(query: str) -> list[Document]:
        """
        Retrieve relevant information from the student knowledge base.

        Use this tool when the user's question requires
        information from documents or contextual knowledge.

        The query should be a clear natural-language search query.
        """

        return execute_rag_query(
            vector_store=vector_store,
            query=query,
        )

    return rag_tool
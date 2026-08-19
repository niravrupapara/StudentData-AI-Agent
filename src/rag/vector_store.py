# src/rag/vector_store.py

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.rag.embedder import embeddings
from src.utils.logger import get_logger


logger = get_logger(__name__)


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Directory for FAISS indexes
INDEX_DIR = BASE_DIR / "storage" / "indexes"


def create_vector_store(
    documents: list[Document],
) -> FAISS:
    """
    Create a FAISS vector store from documents.

    Args:
        documents: Documents to index.

    Returns:
        FAISS vector store.
    """

    if not documents:
        raise ValueError("Cannot create vector store from empty documents.")

    logger.info(
        "Creating FAISS vector store with %d documents.",
        len(documents),
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    logger.info("FAISS vector store created successfully.")

    return vector_store


def save_vector_store(
    vector_store: FAISS,
    index_name: str,
) -> Path:
    """
    Save a FAISS vector store to disk.

    Args:
        vector_store: FAISS vector store.
        index_name: Name of the index directory.

    Returns:
        Path to the saved index.
    """

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / index_name

    vector_store.save_local(str(index_path))

    logger.info(
        "FAISS vector store saved: %s",
        index_path,
    )

    return index_path


def load_vector_store(
    index_name: str,
) -> FAISS:
    """
    Load an existing FAISS vector store from disk.

    Args:
        index_name: Name of the index directory.

    Returns:
        Loaded FAISS vector store.
    """

    index_path = INDEX_DIR / index_name

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    logger.info(
        "Loading FAISS vector store: %s",
        index_path,
    )

    vector_store = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    logger.info("FAISS vector store loaded successfully.")

    return vector_store


def index_exists(index_name: str) -> bool:
    """
    Check whether a FAISS index already exists.
    """

    index_path = INDEX_DIR / index_name

    return (
        index_path.exists()
        and (index_path / "index.faiss").exists()
        and (index_path / "index.pkl").exists()
    )
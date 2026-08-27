import hashlib
import io
from pathlib import Path
from typing import Union

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.ingestion.loaders import load_pdf
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Persistent FAISS directory
FAISS_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "faiss_index"

# Singleton embeddings model (CPU)
_EMBEDDINGS_MODEL: HuggingFaceEmbeddings | None = None

# In-memory cache to avoid repeated disk reads during active session
_FAISS_CACHE: dict = {}


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and cache sentence-transformers embeddings model."""
    global _EMBEDDINGS_MODEL

    if _EMBEDDINGS_MODEL is None:
        logger.info("Initializing HuggingFace embeddings model: all-MiniLM-L6-v2")
        _EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embeddings model loaded successfully.")

    return _EMBEDDINGS_MODEL


def get_or_create_faiss_retriever(file_source: Union[str, Path, io.BytesIO]):
    """Load existing FAISS index from disk or create and persist it for the given PDF."""
    embeddings = get_embeddings()

    # Generate unique hash for this file
    if isinstance(file_source, (str, Path)):
        file_id = hashlib.md5(str(file_source).encode()).hexdigest()
    elif hasattr(file_source, "name"):
        file_id = hashlib.md5(str(file_source.name).encode()).hexdigest()
    else:
        file_id = "default_pdf_index"

    # 1. Check in-memory session cache
    if file_id in _FAISS_CACHE:
        logger.info("Reusing in-memory FAISS index | id=%s", file_id)
        return _FAISS_CACHE[file_id]

    FAISS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = FAISS_STORAGE_DIR / file_id

    # 2. Check disk persistence
    if index_path.exists() and (index_path / "index.faiss").exists():
        logger.info("Loading FAISS index from disk | path=%s", index_path)
        vector_store = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        logger.info("Creating new FAISS vector index on disk | path=%s", index_path)
        text = load_pdf(file_source)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PDF_CHUNK_SIZE,
            chunk_overlap=settings.PDF_CHUNK_OVERLAP,
        )
        documents = splitter.create_documents([text])

        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
        )
        vector_store.save_local(folder_path=str(index_path))
        logger.info("FAISS index successfully saved to disk | path=%s", index_path)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.PDF_RETRIEVAL_K}
    )
    _FAISS_CACHE[file_id] = retriever
    return retriever


@tool
def search_pdf(file_path: str, query: str) -> str:
    """Search an uploaded PDF document using FAISS similarity search and retrieve relevant context text."""
    logger.info("search_pdf tool invoked | file=%s | query=%s", file_path, query)

    try:
        retriever = get_or_create_faiss_retriever(file_path)
        documents = retriever.invoke(query)

        if not documents:
            logger.warning("No relevant content found in PDF for query: %s", query)
            return "No relevant information found in the provided PDF document."

        logger.info("Retrieved %d document chunks from PDF.", len(documents))
        return "\n\n---\n\n".join(doc.page_content for doc in documents)

    except Exception as e:
        logger.exception("Error executing search_pdf tool.")
        return f"Error retrieving context from PDF document: {e}"

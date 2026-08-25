import hashlib
import io
from pathlib import Path
from typing import Union

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.ingestion.pdf_loader import load_pdf
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Persistent storage directory under data/chroma_db
CHROMA_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"

# Singleton embeddings model (loaded once in RAM)
_EMBEDDINGS_MODEL: HuggingFaceEmbeddings | None = None

# In-memory retriever cache to avoid re-opening from disk within the same session
_RETRIEVER_CACHE: dict = {}


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and reuse a single sentence-transformers model instance."""
    global _EMBEDDINGS_MODEL

    if _EMBEDDINGS_MODEL is None:
        logger.info("Initializing SentenceTransformer model: all-MiniLM-L6-v2...")
        _EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("SentenceTransformer model loaded successfully.")

    return _EMBEDDINGS_MODEL


def get_or_create_pdf_retriever(
    file_source: Union[str, Path, io.BytesIO],
    collection_name: str | None = None,
):
    """Load vector store from disk storage or create and persist it if not present."""
    embeddings = get_embeddings()

    # Generate a unique storage directory ID based on file source name/path
    if collection_name:
        file_id = collection_name
    elif isinstance(file_source, (str, Path)):
        file_id = hashlib.md5(str(file_source).encode()).hexdigest()
    elif hasattr(file_source, "name"):
        file_id = hashlib.md5(str(file_source.name).encode()).hexdigest()
    else:
        file_id = "default_pdf_collection"

    # 1. Check in-memory session cache first (instant)
    if file_id in _RETRIEVER_CACHE:
        logger.info("Reusing in-memory PDF retriever | collection=%s", file_id)
        return _RETRIEVER_CACHE[file_id]

    persist_path = str(CHROMA_STORAGE_DIR / file_id)

    # 2. Check disk persistence storage
    if Path(persist_path).exists():
        logger.info("Loading existing Chroma DB from disk storage | path=%s", persist_path)
        vector_store = Chroma(
            collection_name=file_id,
            embedding_function=embeddings,
            persist_directory=persist_path,
        )
    else:
        logger.info("First-time indexing: Creating new Chroma DB on disk | path=%s", persist_path)
        text = load_pdf(file_source)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PDF_CHUNK_SIZE,
            chunk_overlap=settings.PDF_CHUNK_OVERLAP,
        )
        documents = splitter.create_documents([text])

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=file_id,
            persist_directory=persist_path,
        )
        logger.info("Chroma DB successfully persisted to disk | path=%s", persist_path)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.PDF_RETRIEVAL_K}
    )

    _RETRIEVER_CACHE[file_id] = retriever
    return retriever
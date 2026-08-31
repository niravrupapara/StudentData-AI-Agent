import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Persistence paths
INDEX_DIR = Path("data/faiss_index/unified_index")
DOCS_FILE = INDEX_DIR / "registered_docs.json"

_EMBEDDINGS = None
_VECTOR_STORE = None
_REGISTERED_DOCS = set()

def get_embeddings():
    global _EMBEDDINGS
    if not _EMBEDDINGS:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS

def load_store():
    """Load persistent store and registered docs if they exist."""
    global _VECTOR_STORE, _REGISTERED_DOCS
    if _VECTOR_STORE is not None:
        return

    if INDEX_DIR.exists() and (INDEX_DIR / "index.faiss").exists():
        logger.info("Loading persistent FAISS index...")
        _VECTOR_STORE = FAISS.load_local(
            folder_path=str(INDEX_DIR),
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        if DOCS_FILE.exists():
            _REGISTERED_DOCS = set(json.loads(DOCS_FILE.read_text()))
            logger.info("Loaded %d registered docs from persistence.", len(_REGISTERED_DOCS))

def register_document(name: str, text: str, file_type: str, pages: list[str] = None):
    """Chunk document with metadata, add to unified FAISS index, and persist."""
    global _VECTOR_STORE, _REGISTERED_DOCS
    
    load_store() # Ensure store is loaded from disk if app just restarted
    
    if name in _REGISTERED_DOCS: 
        return
    
    logger.info("Registering new document: %s", name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.PDF_CHUNK_SIZE, 
        chunk_overlap=settings.PDF_CHUNK_OVERLAP
    )
    docs = []
    
    if pages:
        for i, page in enumerate(pages, 1):
            if page.strip():
                for chunk in splitter.create_documents([page]):
                    chunk.metadata = {"source": name, "page": i, "file_type": file_type}
                    docs.append(chunk)
    else:
        for chunk in splitter.create_documents([text]):
            chunk.metadata = {"source": name, "page": 0, "file_type": file_type}
            docs.append(chunk)

    if _VECTOR_STORE is None:
        _VECTOR_STORE = FAISS.from_documents(docs, get_embeddings())
    else:
        _VECTOR_STORE.add_documents(docs)
        
    _REGISTERED_DOCS.add(name)
    
    # Save persistent state
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _VECTOR_STORE.save_local(str(INDEX_DIR))
    DOCS_FILE.write_text(json.dumps(list(_REGISTERED_DOCS)))
    
    logger.info("Registered document '%s' | chunks=%d (Persisted to disk)", name, len(docs))

def get_retriever():
    """Return a retriever from the unified FAISS store."""
    load_store()
    return _VECTOR_STORE.as_retriever(search_kwargs={"k": settings.PDF_RETRIEVAL_K}) if _VECTOR_STORE else None

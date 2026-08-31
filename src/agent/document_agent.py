from langchain.agents import create_agent
from langchain_core.tools import tool

from src.llm import get_llm
from src.tools.document_store import get_retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CACHED_AGENT = None

PROMPT = """You are a Document Analysis Agent.

Your job is to answer questions using the uploaded PDF and TXT documents.

You have one tool:
- retrieve_chunks(query): searches the uploaded document index.

Instructions:
1. Search the documents before answering.
2. You may call retrieve_chunks multiple times if the first search is insufficient.
3. Use only information found in the retrieved documents.
4. Do not invent information.
5. Always cite the source filename and page number when available.
6. If the documents do not contain enough information, clearly say so.
"""


@tool
def retrieve_chunks(query: str) -> str:
    """Search all uploaded PDF and TXT documents for relevant content."""

    retriever = get_retriever()

    if retriever is None:
        return "No documents have been uploaded yet."

    try:
        docs = retriever.invoke(query)
    except Exception as e:
        logger.exception("Document retrieval failed: %s", e)
        return f"Document retrieval failed: {e}"

    if not docs:
        return "No relevant content found in the uploaded documents."

    parts = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")

        if page is not None:
            header = f"[Source: {source}, Page: {page}]"
        else:
            header = f"[Source: {source}]"

        parts.append(
            f"{header}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


def get_document_agent():
    """Return the cached Document Agent, creating it only when necessary."""

    global _CACHED_AGENT

    if _CACHED_AGENT is not None:
        return _CACHED_AGENT

    if get_retriever() is None:
        return None

    logger.info("Building Document Agent...")

    _CACHED_AGENT = create_agent(
        model=get_llm(),
        tools=[retrieve_chunks],
        system_prompt=PROMPT,
    )

    logger.info("Document Agent created successfully.")

    return _CACHED_AGENT


def invalidate_agent():
    """Clear the cached agent.

    Call this only when the agent configuration changes.
    """

    global _CACHED_AGENT
    _CACHED_AGENT = None


@tool
def search_documents(query: str) -> str:
    """Use the Document Agent to answer questions about uploaded PDF/TXT files."""

    agent = get_document_agent()

    if agent is None:
        return "No PDF or TXT documents have been uploaded yet."

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            }
        )

        messages = result.get("messages", [])

        if not messages:
            return "No response generated."

        return messages[-1].content

    except Exception as e:
        logger.exception("Document Agent failed: %s", e)
        return f"Error searching documents: {e}"
from pathlib import Path

from src.agent.pdf_retriever import get_or_create_pdf_retriever
from src.llm import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pdf_agent(
    file_source: str | Path,
    user_query: str,
) -> str:
    """Answer a question using information retrieved from a PDF."""

    logger.info(
        "Running PDF agent | file=%s | query=%s",
        file_source,
        user_query,
    )

    try:
        retriever = get_or_create_pdf_retriever(file_source)

        logger.info("Retrieving relevant PDF content...")

        documents = retriever.invoke(user_query)

        if not documents:
            logger.warning(
                "No relevant PDF content found | query=%s",
                user_query,
            )

            return (
                "I could not find relevant information "
                "in the PDF."
            )

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        logger.info(
            "Retrieved PDF documents | count=%d",
            len(documents),
        )

        llm = get_llm()

        prompt = f"""
You are a PDF question-answering agent.

Answer the user's question using ONLY the information
provided in the retrieved PDF context.

If the answer cannot be found in the context, clearly say
that the information is not available in the PDF.

Do not invent or assume information.

Retrieved PDF context:
{context}

User question:
{user_query}
"""

        response = llm.invoke(prompt)

        logger.info("PDF agent completed successfully.")

        return response.content

    except Exception:
        logger.exception("PDF agent execution failed.")
        raise
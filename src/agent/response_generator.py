# src/agent/response_generator.py

from src.agent.state import AgentState
from src.llm.client import llm
from src.llm.prompts import (
    RESPONSE_SYSTEM_PROMPT,
    build_response_prompt,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def generate_response(state: AgentState) -> dict:
    """
    Generate the final response using the user query and available context.
    """

    user_query = state["user_query"]
    intent = state.get("intent", "DIRECT")

    logger.info(
        "Generating final response | intent=%s",
        intent,
    )

    try:
        # ------------------------------------------------------------
        # DIRECT query
        # ------------------------------------------------------------

        if intent == "DIRECT":
            context = ""

        # ------------------------------------------------------------
        # DATA query
        # ------------------------------------------------------------

        else:
            context = _build_data_context(state)

        prompt = build_response_prompt(
            user_query=user_query,
            context=context,
        )

        response = llm.invoke(
            [
                ("system", RESPONSE_SYSTEM_PROMPT),
                ("human", prompt),
            ]
        )

        final_answer = response.content.strip()

        logger.info("Final response generated successfully.")

        return {
            "final_answer": final_answer,
        }

    except Exception:
        logger.exception("Failed to generate final response.")
        raise


def _build_data_context(state: AgentState) -> str:
    """
    Combine Pandas and RAG results into a single context.
    """

    pandas_result = state.get("pandas_result")
    rag_result = state.get("rag_result", [])

    context_parts = []

    # Pandas result
    if pandas_result is not None:
        context_parts.append(
            "PANDAS RESULT:\n"
            f"{pandas_result}"
        )

    # RAG results
    if rag_result:
        rag_documents = []

        for document in rag_result:
            rag_documents.append(
                document.page_content
            )

        context_parts.append(
            "RAG RESULTS:\n"
            + "\n\n".join(rag_documents)
        )

    if not context_parts:
        return "No relevant data was found."

    return "\n\n".join(context_parts)

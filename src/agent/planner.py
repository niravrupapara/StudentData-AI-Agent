# src/agent/planner.py

from src.agent.state import AgentState
from src.llm.client import llm
from src.llm.prompts import INTENT_PLANNER_SYSTEM_PROMPT
from src.utils.logger import get_logger


logger = get_logger(__name__)


VALID_INTENTS = {"DIRECT", "DATA"}


def plan_intent(state: AgentState) -> dict:
    """
    Classify the user query as DIRECT or DATA.
    """

    user_query = state["user_query"].strip()

    if not user_query:
        raise ValueError("User query cannot be empty.")

    logger.info("Planning user query: %s", user_query)

    try:
        response = llm.invoke(
            [
                ("system", INTENT_PLANNER_SYSTEM_PROMPT),
                ("human", user_query),
            ]
        )

        intent = response.content.strip().upper()

        # Handle accidental formatting from the LLM.
        intent = intent.replace("`", "").strip()

        if intent not in VALID_INTENTS:
            logger.warning(
                "Invalid planner output: %s. Falling back to DATA.",
                intent,
            )
            intent = "DATA"

        logger.info("Query intent classified as: %s", intent)

        return {
            "intent": intent,
        }

    except Exception:
        logger.exception("Failed to classify user query.")
        raise
# src/agent/nodes.py

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.utils.logger import get_logger
from src.utils.message_utils import format_conversation_history, get_current_user_query


logger = get_logger(__name__)


def execute_pandas_agent(state: AgentState, pandas_agent) -> dict:
    """Execute the Pandas Agent with conversation history context."""
    messages = state["messages"]
    logger.info("Number of messages in state: %d", len(messages))

    history = format_conversation_history(messages)
    user_query = get_current_user_query(messages)
    logger.info("Current user query: %s", user_query)

    # If previous conversation exists, prepend it. Otherwise, send just the query.
    if history.strip():
        agent_input = (
            f"Previous conversation:\n{history}\n\n"
            f"Current question:\n{user_query}"
        )
        logger.info("Conversation history included in prompt.")
    else:
        agent_input = user_query
        logger.info("No prior history - sending direct user query.")

    logger.info("Input sent to Pandas agent:\n%s", agent_input)

    try:
        response = pandas_agent.invoke({"input": agent_input})
        output_text = response.get("output", "No response generated.")
        return {"messages": [AIMessage(content=output_text)]}

    except Exception:
        logger.exception("Failed to execute Pandas agent.")
        raise



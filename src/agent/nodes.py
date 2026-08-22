# src/agent/nodes.py

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.state import AgentState
from src.utils.logger import get_logger


logger = get_logger(__name__)


def execute_pandas_agent(state: AgentState, pandas_agent) -> dict:

    """Execute the Pandas Agent using the latest user message from state."""

    last_message = state["messages"][-1]
    user_query = (
        last_message.content
        if isinstance(last_message, HumanMessage)
        else str(last_message)
    )

    logger.info("Node executing query: %s", user_query)

    try:
        response = pandas_agent.invoke({"input": user_query})
        output_text = response.get("output", "No response generated.")
        return {"messages": [AIMessage(content=output_text)]}

    except Exception:
        logger.exception("Failed to execute Pandas agent.")
        raise


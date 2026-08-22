# src/agent/runner.py

from langchain_core.messages import HumanMessage

from src.utils.logger import get_logger


logger = get_logger(__name__)


def ask_agent(
    graph,
    question: str,
    thread_id: str = "default_session",
) -> str:
    """Submit a question to the compiled graph with a session thread_id."""
    logger.info(
        "Running query | thread_id=%s | question=%s",
        thread_id,
        question,
    )

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    last_message = result["messages"][-1]
    return last_message.content

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.supervisor import (
    build_supervisor_messages,
    get_supervisor,
)
from src.agent.supervisor_tools import SUPERVISOR_TOOLS
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_ITERATIONS = 8


def supervisor_node(state: AgentState) -> dict:
    """Run one Supervisor reasoning step."""
    iteration_count = state.get("iteration_count", 0) + 1

    logger.info(
        "Supervisor iteration started | iteration=%d",
        iteration_count,
    )

    if iteration_count > MAX_ITERATIONS:
        logger.warning("Supervisor reached maximum iterations.")
        return {
            "error": "Maximum Supervisor iterations reached."
        }

    messages = state.get("messages", [])
    files = state.get("files", [])

    if not messages:
        raise ValueError("No messages found in agent state.")

    supervisor_messages = build_supervisor_messages(
        messages=messages,
        files=files,
    )

    supervisor = get_supervisor()

    try:
        response = supervisor.invoke(supervisor_messages)

        logger.info(
            "Supervisor response received | tool_calls=%d",
            len(response.tool_calls) if hasattr(response, "tool_calls") else 0,
        )

        return {
            "messages": [response],
            "iteration_count": iteration_count,
        }

    except Exception:
        logger.exception("Supervisor execution failed.")
        raise


def route_supervisor(state: AgentState) -> str:
    """Determine whether Supervisor wants to call a tool or finish."""
    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No messages available for routing.")

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        logger.info("Supervisor requested tool execution.")
        return "tools"

    logger.info("Supervisor completed reasoning.")
    return END


def build_agent_graph(checkpointer=None):
    """Build and compile the multi-agent Supervisor graph with state memory."""
    logger.info("Building multi-agent Supervisor graph...")

    tool_node = ToolNode(SUPERVISOR_TOOLS)

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "supervisor")

    memory = checkpointer if checkpointer is not None else MemorySaver()
    compiled_graph = builder.compile(checkpointer=memory)

    logger.info("Multi-agent graph compiled successfully with MemorySaver.")
    return compiled_graph


def ask_agent(
    graph,
    question: str,
    files: list[str] | None = None,
    thread_id: str = "default_session",
) -> str:
    """Submit a question to the compiled multi-agent graph with session thread_id and files."""
    logger.info(
        "Running query | thread_id=%s | files=%s | question=%s",
        thread_id,
        files or [],
        question,
    )

    config = {"configurable": {"thread_id": thread_id}}

    payload = {
        "messages": [HumanMessage(content=question)],
        "files": files or [],
    }

    result = graph.invoke(
        payload,
        config=config,
    )

    last_message = result["messages"][-1]
    return getattr(last_message, "content", str(last_message))

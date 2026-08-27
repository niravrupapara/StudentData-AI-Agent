from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.agent.supervisor import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_TOOLS
from src.llm import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_agent_graph(checkpointer=None):
    """Build the official LangGraph ReAct Agent with tool calling and conversation memory."""
    logger.info("Initializing LangGraph ReAct Agent...")
    llm = get_llm()
    memory = checkpointer if checkpointer is not None else MemorySaver()

    # Prebuilt LangGraph ReAct Agent
    agent = create_react_agent(
        model=llm,
        tools=SUPERVISOR_TOOLS,
        prompt=SUPERVISOR_SYSTEM_PROMPT,
        checkpointer=memory,
    )

    logger.info("ReAct Agent compiled successfully with MemorySaver.")
    return agent


def ask_agent(
    graph,
    question: str,
    files: list[str] | None = None,
    thread_id: str = "default_session",
) -> str:
    """Submit a question to the ReAct agent with conversation thread and file context."""
    logger.info("ask_agent invoked | thread_id=%s | question=%s", thread_id, question)

    config = {"configurable": {"thread_id": thread_id}}

    if files:
        file_list = "\n".join(f"- {f}" for f in files)
        formatted_question = f"Available Files:\n{file_list}\n\nUser Question:\n{question}"
    else:
        formatted_question = question

    result = graph.invoke(
        {"messages": [HumanMessage(content=formatted_question)]},
        config=config,
    )

    last_message = result["messages"][-1]
    return getattr(last_message, "content", str(last_message))
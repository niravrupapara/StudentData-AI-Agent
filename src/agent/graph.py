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
) -> dict:
    """Submit a question to the ReAct agent with conversation thread and file context."""
    logger.info("ask_agent invoked | thread_id=%s | question=%s", thread_id, question)

    config = {"configurable": {"thread_id": thread_id}}

    if files:
        formatted_files = []
        for f in files:
            from pathlib import Path
            suffix = Path(f).suffix.lower()
            if suffix in [".csv", ".xlsx", ".xls"]:
                tag = "Tabular Data (use analyze_data)"
            elif suffix == ".pdf":
                tag = "PDF Document (use search_pdf)"
            else:
                tag = "Document/File"
            formatted_files.append(f"- {f} [{tag}]")
        file_list = "\n".join(formatted_files)
        formatted_question = f"Available Files:\n{file_list}\n\nUser Question:\n{question}"
    else:
        formatted_question = question

    result = graph.invoke(
        {"messages": [HumanMessage(content=formatted_question)]},
        config=config,
    )

    messages = result["messages"]
    last_message = messages[-1]
    content = getattr(last_message, "content", str(last_message))

    # Extract image artifacts from ToolMessages produced in this turn
    artifacts = []
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            last_human_idx = i
            break

    turn_messages = messages[last_human_idx:] if last_human_idx != -1 else messages
    for msg in turn_messages:
        artifact = getattr(msg, "artifact", None)
        if isinstance(artifact, dict) and artifact.get("type") == "image":
            artifacts.append(artifact)

    logger.info("ask_agent execution complete | artifacts_count=%d | artifacts=%s", len(artifacts), artifacts)
    return {"content": content, "artifacts": artifacts}
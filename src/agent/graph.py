# src/agent/graph.py

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.llm.client import llm
from src.llm.prompts import TOOL_CALLING_SYSTEM_PROMPT
from src.tools.pandas_tool import create_pandas_tool
from src.tools.rag_tool import create_rag_tool
from src.utils.logger import get_logger


logger = get_logger(__name__)


def build_student_agent_graph(
    dataframe,
    schema,
    vector_store,
):
    """
    Build the optimized Student Data Agent graph.

    The LLM decides whether to:
    - answer directly,
    - call Pandas,
    - call RAG.
    """

    logger.info("Building Student Data Agent graph.")

    # ------------------------------------------------------------
    # Create tools once
    # ------------------------------------------------------------

    pandas_tool = create_pandas_tool(dataframe)
    rag_tool = create_rag_tool(vector_store)

    tools = [
        pandas_tool,
        rag_tool,
    ]

    # Bind tools to the LLM once.
    tool_llm = llm.bind_tools(tools)

    # ------------------------------------------------------------
    # Build system prompt with current dataset metadata
    # ------------------------------------------------------------

    system_prompt = TOOL_CALLING_SYSTEM_PROMPT.format(
        schema=schema,
    )

    # ------------------------------------------------------------
    # LLM node
    # ------------------------------------------------------------

    def call_llm(state: AgentState) -> dict:
        """
        Call the LLM with the current dataset schema
        and available tools.
        """

        logger.info("Calling main LLM.")

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            *state["messages"],
        ]

        response = tool_llm.invoke(messages)

        logger.info(
            "LLM completed | tool_calls=%d",
            len(getattr(response, "tool_calls", [])),
        )

        return {
            "messages": [response],
        }

    # ------------------------------------------------------------
    # Tool node
    # ------------------------------------------------------------

    tool_node = ToolNode(tools)

    # ------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------

    def route_after_llm(state: AgentState) -> str:
        """
        Route to tools only when the LLM requests a tool.
        Otherwise finish.
        """

        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"

        return "end"

    # ------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------

    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "llm")

    graph.add_conditional_edges(
        "llm",
        route_after_llm,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # Tool result → LLM
    graph.add_edge("tools", "llm")

    compiled_graph = graph.compile()

    logger.info(
        "Student Data Agent graph built successfully."
    )

    return compiled_graph


def run_agent_query(
    graph,
    user_query: str,
) -> str:
    """
    Run a user query through the Student Data Agent.
    """

    logger.info(
        "Running agent query: %s",
        user_query,
    )

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query,
                }
            ]
        }
    )

    messages = result.get("messages", [])

    if not messages:
        return ""

    final_message = messages[-1]

    final_answer = final_message.content

    logger.info(
        "Agent query completed successfully."
    )

    return final_answer
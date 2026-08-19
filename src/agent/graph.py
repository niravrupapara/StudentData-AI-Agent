# src/agent/graph.py

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.planner import plan_intent
from src.agent.response_generator import generate_response
from src.agent.state import AgentState
from src.tools.pandas_tool import execute_pandas_query
from src.tools.rag_tool import execute_rag_query
from src.agent.pandas_query_generator import generate_pandas_query

from src.utils.logger import get_logger


logger = get_logger(__name__)


def _run_data_tools(state: AgentState) -> dict:
    """
    Execute both Pandas and RAG for a DATA query.
    """

    logger.info("Executing DATA pipeline: Pandas + RAG")

    dataframe = state.get("dataframe")
    vector_store = state.get("vector_store")
    user_query = state["user_query"]

    if dataframe is None:
        raise ValueError("DataFrame is not available.")

    if vector_store is None:
        raise ValueError("Vector store is not available.")

    try:
        # ------------------------------------------------------------
        # Pandas
        # ------------------------------------------------------------

        pandas_query = _build_pandas_query(
            user_query=user_query,
            schema=state.get("schema", {}),
        )

        pandas_result = execute_pandas_query(
            dataframe=dataframe,
            query=pandas_query,
        )

        # ------------------------------------------------------------
        # RAG
        # ------------------------------------------------------------

        rag_result = execute_rag_query(
            vector_store=vector_store,
            query=user_query,
        )

        logger.info("DATA pipeline completed successfully.")

        return {
            "pandas_result": pandas_result,
            "rag_result": rag_result,
        }

    except Exception:
        logger.exception("DATA pipeline failed.")
        raise


def _build_pandas_query(
    user_query: str,
    schema: dict[str, Any],
) -> str:
    return generate_pandas_query(
        user_query=user_query,
        schema=schema,
    )

def _route_after_planner(state: AgentState) -> str:
    """
    Decide which graph node should run after intent planning.
    """

    if state.get("intent") == "DATA":
        return "data_tools"

    return "response"


def build_student_agent_graph():
    """
    Build and compile the Student Data Agent LangGraph.
    """

    logger.info("Building Student Data Agent graph.")

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("planner", plan_intent)
    graph.add_node("data_tools", _run_data_tools)
    graph.add_node("response", generate_response)

    # Entry
    graph.add_edge(START, "planner")

    # Conditional routing
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "data_tools": "data_tools",
            "response": "response",
        },
    )

    # Final response
    graph.add_edge("data_tools", "response")
    graph.add_edge("response", END)

    compiled_graph = graph.compile()

    logger.info("Student Data Agent graph built successfully.")

    return compiled_graph


def run_agent_query(
    graph,
    user_query: str,
    dataframe,
    schema,
    vector_store,
) -> str:
    """
    Run a user query through the agent graph.
    """

    logger.info("Running agent query: %s", user_query)

    initial_state: AgentState = {
        "user_query": user_query,
        "dataframe": dataframe,
        "schema": schema,
        "vector_store": vector_store,
    }

    result = graph.invoke(initial_state)

    final_answer = result.get("final_answer", "")

    logger.info("Agent query completed successfully.")

    return final_answer
# src/agent/graph.py

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import execute_pandas_agent
from src.agent.pandas_agent import create_pandas_agent
from src.agent.state import AgentState
from src.utils.logger import get_logger


logger = get_logger(__name__)


def build_agent_graph(df: pd.DataFrame):
    """Build and compile the LangGraph StateGraph with MemorySaver."""
    logger.info("Building LangGraph StateGraph...")

    # 1. Create underlying Pandas Agent
    pandas_agent = create_pandas_agent(df)

    # 2. Node calls execute_pandas_agent directly
    def agent_node(state: AgentState) -> dict:
        return execute_pandas_agent(state, pandas_agent)

    # 3. Assemble and Compile Graph
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    checkpointer = MemorySaver()
    compiled_graph = builder.compile(checkpointer=checkpointer)

    logger.info("StateGraph compiled successfully.")
    return compiled_graph



from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Conversation and execution state for the LangGraph agent workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
    files: list[str]
    iteration_count: int
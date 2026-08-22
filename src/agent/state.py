# src/agent/state.py

from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Conversation state with automatic history reducer."""

    messages: Annotated[list[AnyMessage], add_messages]



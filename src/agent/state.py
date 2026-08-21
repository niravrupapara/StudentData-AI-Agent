# src/agent/state.py

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    """
    State shared between LangGraph nodes.
    """

    # Conversation / tool-calling messages
    messages: Annotated[list[AnyMessage], add_messages]

    # Original user question
    user_query: str

    # Loaded student DataFrame
    dataframe: Any

    # Dataset schema information
    schema: dict[str, Any]

    # FAISS vector store
    vector_store: Any
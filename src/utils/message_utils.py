from langchain_core.messages import AIMessage, HumanMessage, AnyMessage


def format_conversation_history(
    messages: list[AnyMessage],
    max_messages: int = 4,
) -> str:
    """Format the most recent previous messages into readable conversation text."""

    # Exclude the current/latest message.
    previous_messages = messages[:-1]

    # Keep only the most recent N previous messages.
    previous_messages = previous_messages[-max_messages:]

    history = []

    for message in previous_messages:
        if isinstance(message, HumanMessage):
            history.append(f"User: {message.content}")

        elif isinstance(message, AIMessage):
            history.append(f"Assistant: {message.content}")

    return "\n".join(history)


def get_current_user_query(messages: list[AnyMessage]) -> str:
    """Get the latest user query from the message history."""

    last_message = messages[-1]

    if isinstance(last_message, HumanMessage):
        return last_message.content

    return str(last_message)
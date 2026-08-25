from langchain_core.messages import HumanMessage

from src.agent.graph import graph


state = {
    "messages": [
        HumanMessage(
            content="Which department has the highest average marks?"
        )
    ],
    "files": [
        "data/student.pdf"
    ],
    "iteration_count": 0,
}


result = graph.invoke(state)

print("\n--- FINAL RESULT ---\n")

for message in result["messages"]:
    print(f"{type(message).__name__}:")
    print(message.content)
    print()
# tests/test_agent.py

import os
from pathlib import Path
from dotenv import load_dotenv

from src.agent.graph import ask_agent, build_agent_graph
from src.ingestion.csv_loader import load_csv
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def run_tests():
    """Run CLI test queries on a sample or provided CSV file."""

    csv_path = os.getenv(
        "TEST_CSV_PATH",
        r"C:\Users\Nirav Rupapara\Downloads\student_schedule_synthetic_1250.csv",
    )

    if not Path(csv_path).exists():
        print(f"Warning: Test CSV path does not exist: {csv_path}")
        print("Please set TEST_CSV_PATH in .env or provide a valid CSV file path.")
        return

    print("=" * 70)
    print("1. Validating CSV...")
    print("=" * 70)

    df = load_csv(csv_path)

    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    print("\n" + "=" * 70)
    print("2. Initializing LangGraph Multi-Agent System...")
    print("=" * 70)

    graph = build_agent_graph()

    questions = [
        "How many students are from each branch?",
        "What is machine learning?",
    ]

    print("\n" + "=" * 70)
    print("3. Executing Test Queries with Multi-Agent Memory...")
    print("=" * 70)

    for question in questions:
        print("\n" + "-" * 50)
        print(f"QUESTION: {question}")
        print("-" * 50)

        try:
            answer = ask_agent(
                graph=graph,
                question=question,
                files=[str(csv_path)],
                thread_id="cli_test_session",
            )
            print(f"ANSWER:\n{answer}")
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    run_tests()


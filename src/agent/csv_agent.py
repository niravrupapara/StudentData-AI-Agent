from pathlib import Path

from src.agent.pandas_agent import create_pandas_agent
from src.ingestion.csv_loader import load_csv
from src.utils.logger import get_logger


logger = get_logger(__name__)


def run_csv_agent(file_source: str | Path, user_query: str) -> str:
    """Run the Pandas agent against a CSV file."""

    logger.info(
        "Running CSV agent | file=%s | query=%s",
        file_source,
        user_query,
    )

    try:
        df = load_csv(file_source)

        agent = create_pandas_agent(df)

        response = agent.invoke(
            {
                "input": user_query,
            }
        )

        output = response.get(
            "output",
            "No response generated.",
        )

        logger.info("CSV agent completed successfully.")

        return output

    except Exception:
        logger.exception("CSV agent execution failed.")
        raise
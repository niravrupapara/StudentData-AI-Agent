from pathlib import Path
from typing import Sequence, Union

from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

from src.ingestion.loaders import load_csv, load_excel
from src.llm import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_pandas_agent(df: Union[pd.DataFrame, Sequence[pd.DataFrame]]):
    """Create a LangChain Pandas DataFrame Agent for single or multiple DataFrames."""
    logger.info("Initializing Pandas DataFrame Sub-Agent...")
    llm = get_llm()

    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True,
    )


def run_pandas_agent(file_path: Union[str, Path], user_query: str) -> str:
    """Load CSV or Excel file into DataFrame(s) and query with Pandas Sub-Agent."""
    logger.info("Running Pandas analysis | file=%s | query=%s", file_path, user_query)
    p = Path(file_path)

    try:
        # Load file into DataFrame(s) based on extension
        if p.suffix.lower() == ".csv":
            df_data = load_csv(p)
        elif p.suffix.lower() in [".xlsx", ".xls"]:
            sheets = load_excel(p)
            df_data = list(sheets.values()) if len(sheets) > 1 else list(sheets.values())[0]
        else:
            raise ValueError(f"Unsupported tabular data format: {p.suffix}")

        # Run Pandas agent
        agent = create_pandas_agent(df_data)
        response = agent.invoke({"input": user_query})
        output = response.get("output", "No response generated.")
        logger.info("Pandas analysis completed successfully.")
        return output

    except Exception:
        logger.exception("Pandas analysis execution failed.")
        raise


@tool
def analyze_data(file_path: str, query: str) -> str:
    """Analyze a CSV or Excel file using the Pandas DataFrame Agent to answer data and aggregation questions."""
    logger.info("Tool analyze_data called | file=%s | query=%s", file_path, query)
    return run_pandas_agent(file_path=file_path, user_query=query)

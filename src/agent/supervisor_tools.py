from pathlib import Path

from langchain_core.tools import tool

from src.agent.csv_agent import run_csv_agent
from src.agent.excel_agent import run_excel_agent
from src.agent.pdf_agent import run_pdf_agent
from src.agent.visualization_tool import generate_chart
from src.utils.logger import get_logger

logger = get_logger(__name__)


@tool
def analyze_csv(file_path: str, query: str) -> str:
    """Analyze a CSV file and answer a data-related question."""

    logger.info(
        "Supervisor requested CSV analysis | file=%s | query=%s",
        file_path,
        query,
    )

    return run_csv_agent(
        file_source=Path(file_path),
        user_query=query,
    )


@tool
def analyze_excel(file_path: str, query: str) -> str:
    """Analyze an Excel workbook and answer a data-related question."""

    logger.info(
        "Supervisor requested Excel analysis | file=%s | query=%s",
        file_path,
        query,
    )

    return run_excel_agent(
        file_source=Path(file_path),
        user_query=query,
    )


@tool
def search_pdf(file_path: str, query: str) -> str:
    """Search a PDF and answer a question using its content."""

    logger.info(
        "Supervisor requested PDF analysis | file=%s | query=%s",
        file_path,
        query,
    )

    return run_pdf_agent(
        file_source=Path(file_path),
        user_query=query,
    )


SUPERVISOR_TOOLS = [
    analyze_csv,
    analyze_excel,
    search_pdf,
    generate_chart,
]
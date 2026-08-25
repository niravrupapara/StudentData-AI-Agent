from langchain_core.messages import SystemMessage

from src.agent.supervisor_tools import SUPERVISOR_TOOLS
from src.llm import get_llm
from src.utils.logger import get_logger


logger = get_logger(__name__)


SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor Agent of a multi-agent data analysis and visualization system.

Your responsibility is to understand the user's request, decide what
work is required, and delegate that work to the appropriate tools.

Available capabilities:

- analyze_csv:
  Analyze CSV files using a dataframe analysis agent.

- analyze_excel:
  Analyze Excel workbooks, including their sheets.

- search_pdf:
  Search PDF documents and answer questions using their content.

- generate_chart:
  Execute Python matplotlib/seaborn code to generate and save chart figures.

Rules:

1. Do not perform data analysis yourself when a specialized tool can do it.
2. Use the appropriate tool for the file type.
3. When the user requests a chart, plot, graph, or visual distribution:
   a. First, call `analyze_csv` or `analyze_excel` to extract the required data and aggregates.
   b. Then, call `generate_chart` with valid Python matplotlib/seaborn code using the extracted data.
   c. Include the saved chart path in your answer if reported by the tool.
4. You may call multiple tools when the request requires information from multiple files or steps.
5. Use the exact file paths provided in the available files list.
6. When all required work is complete, provide a clear, concise final answer.
"""


def get_supervisor():
    """Create and return the configured Supervisor LLM."""

    logger.info("Creating Supervisor Agent...")

    try:
        llm = get_llm()

        supervisor = llm.bind_tools(
            SUPERVISOR_TOOLS
        )

        logger.info(
            "Supervisor Agent created successfully | tools=%d",
            len(SUPERVISOR_TOOLS),
        )

        return supervisor

    except Exception:
        logger.exception(
            "Failed to create Supervisor Agent."
        )
        raise


def build_supervisor_messages(
    messages,
    files: list[str],
):
    """Build messages containing conversation and available files."""

    file_context = (
        "Available files:\n"
        + "\n".join(f"- {file}" for file in files)
    )

    return [
        SystemMessage(
            content=SUPERVISOR_SYSTEM_PROMPT
            + "\n\n"
            + file_context
        ),
        *messages,
    ]
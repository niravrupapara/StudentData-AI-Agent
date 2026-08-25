from pathlib import Path

from src.agent.pandas_agent import create_pandas_agent
from src.ingestion.excel_loader import load_excel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_excel_agent(
    file_source: str | Path,
    user_query: str,
) -> str:
    """Run analysis on an Excel workbook across all its sheets."""

    logger.info(
        "Running Excel agent | file=%s | query=%s",
        file_source,
        user_query,
    )

    try:
        sheets = load_excel(file_source)
        sheet_names = list(sheets.keys())

        logger.info(
            "Excel workbook loaded | sheet_count=%d | sheets=%s",
            len(sheets),
            sheet_names,
        )

        # Build sheet descriptions and list of DataFrames
        sheet_descriptions = []
        df_list = []
        for i, (name, df) in enumerate(sheets.items(), start=1):
            sheet_descriptions.append(
                f"- df{i}: sheet '{name}' (Columns: {list(df.columns)})"
            )
            df_list.append(df)

        sheet_context = "\n".join(sheet_descriptions)

        # If single sheet, pass df directly; if multi-sheet, pass df_list
        target_df = df_list[0] if len(df_list) == 1 else df_list
        agent = create_pandas_agent(target_df)

        enhanced_input = (
            f"Available Excel sheets:\n{sheet_context}\n\n"
            f"User Question: {user_query}"
        )

        response = agent.invoke({"input": enhanced_input})
        output = response.get("output", "No response generated.")

        logger.info("Excel agent completed successfully.")
        return output

    except Exception:
        logger.exception("Excel agent execution failed.")
        raise
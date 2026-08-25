import io
from pathlib import Path
from typing import Union

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

FileSource = Union[str, Path, io.BytesIO]


def load_excel(file_source: FileSource) -> dict[str, pd.DataFrame]:
    """Load and validate all sheets from an Excel workbook into a dictionary of DataFrames."""
    logger.info("Loading Excel data source: %s", file_source)

    try:
        sheets = pd.read_excel(
            file_source,
            sheet_name=None,
        )

        if not sheets:
            logger.error("Excel file contains no sheets.")
            raise ValueError("The provided Excel file contains no data.")

        logger.info(
            "Excel loaded successfully | sheets_count=%d | sheet_names=%s",
            len(sheets),
            list(sheets.keys()),
        )

        for sheet_name, df in sheets.items():
            logger.info(
                "Excel sheet parsed | sheet=%s | rows=%d | columns=%d",
                sheet_name,
                len(df),
                len(df.columns),
            )

        return sheets

    except Exception as e:
        logger.exception("Failed to load Excel file.")
        raise ValueError(f"Error loading Excel file: {e}") from e

# src/data_loader.py

from pathlib import Path
from typing import Union
import io
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_csv(file_source: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
    """
    Load a CSV file from a file path or file-like buffer into a DataFrame.

    Args:
        file_source: File path (str/Path) or file-like buffer (e.g. from Streamlit).

    Returns:
        Loaded pandas DataFrame.

    Raises:
        ValueError: If the CSV is empty or cannot be read.
    """
    logger.info("Loading CSV data source...")

    try:
        df = pd.read_csv(file_source)

        if df.empty:
            logger.error("Loaded CSV file is empty.")
            raise ValueError("The provided CSV file contains no data.")

        logger.info(
            "CSV loaded successfully | rows=%d | columns=%d",
            len(df),
            len(df.columns),
        )

        return df

    except Exception:
        logger.exception("Failed to load CSV file.")
        raise

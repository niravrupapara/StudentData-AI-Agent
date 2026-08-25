import io
from pathlib import Path
from typing import Union

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

FileSource = Union[str, Path, io.BytesIO]


def load_csv(file_source: FileSource) -> pd.DataFrame:
    """Load and validate a CSV data source into a Pandas DataFrame."""
    logger.info("Loading CSV data source: %s", file_source)

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

    except Exception as e:
        logger.exception("Failed to load CSV file.")
        raise ValueError(f"Error loading CSV file: {e}") from e

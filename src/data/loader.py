# src/data/loader.py

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_csv_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded Pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is not a CSV or cannot be loaded.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("CSV file not found: %s", file_path)
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        logger.error("Invalid file type: %s", file_path)
        raise ValueError("Only CSV files are supported.")

    try:
        logger.info("Loading CSV file: %s", file_path)

        dataframe = pd.read_csv(file_path)

        logger.info(
            "CSV loaded successfully | rows=%d | columns=%d",
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe

    except Exception:
        logger.exception("Failed to load CSV file: %s", file_path)
        raise
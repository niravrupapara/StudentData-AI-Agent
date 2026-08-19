# src/data/schema.py

from typing import Any

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


def extract_schema_info(dataframe: pd.DataFrame) -> dict[str, Any]:
    """
    Extract useful schema information from a DataFrame.

    Args:
        dataframe: Input Pandas DataFrame.

    Returns:
        Dictionary containing dataset structure information.
    """

    if dataframe.empty:
        logger.warning("Schema requested for an empty DataFrame.")

    schema = {
        "columns": dataframe.columns.tolist(),
        "dtypes": {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "sample_rows": dataframe.head(5).to_dict(orient="records"),
    }

    logger.info(
        "Schema extracted | rows=%d | columns=%d",
        schema["row_count"],
        schema["column_count"],
    )

    return schema
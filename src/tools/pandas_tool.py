# src/tools/pandas_tool.py

from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.utils.logger import get_logger


logger = get_logger(__name__)


def execute_pandas_query(
    dataframe: pd.DataFrame,
    query: str,
) -> Any:
    """
    Execute a Pandas expression on the provided DataFrame.
    """

    if dataframe.empty:
        logger.warning(
            "Pandas query received an empty DataFrame."
        )
        return None

    if not query or not query.strip():
        raise ValueError(
            "Pandas query cannot be empty."
        )

    logger.info(
        "Executing Pandas query: %s",
        query,
    )

    try:
        local_namespace = {
            "df": dataframe,
            "pd": pd,
        }

        result = eval(
            query,
            {"__builtins__": {}},
            local_namespace,
        )

        logger.info(
            "Pandas query executed successfully."
        )

        return result

    except Exception:
        logger.exception(
            "Pandas query execution failed: %s",
            query,
        )
        raise


def create_pandas_tool(
    dataframe: pd.DataFrame,
):
    """
    Create a Pandas tool bound to the application DataFrame.
    """

    @tool
    def pandas_tool(query: str) -> Any:
        """
        Execute a Pandas expression on the student DataFrame.

        Use this tool for structured-data questions such as:
        filtering, counting, sorting, grouping, aggregation,
        and finding maximum or minimum values.

        The query must be a valid Pandas expression using `df`.
        """

        return execute_pandas_query(
            dataframe=dataframe,
            query=query,
        )

    return pandas_tool
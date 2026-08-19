# src/rag/document_builder.py

from typing import Any

import pandas as pd
from langchain_core.documents import Document

from src.utils.logger import get_logger


logger = get_logger(__name__)


def dataframe_to_documents(
    dataframe: pd.DataFrame,
) -> list[Document]:
    """
    Convert each DataFrame row into a LangChain Document.

    Each row becomes one independent document.

    Args:
        dataframe: Input student DataFrame.

    Returns:
        List of LangChain Document objects.
    """

    if dataframe.empty:
        logger.warning("Cannot build documents from an empty DataFrame.")
        return []

    documents = []

    for row_index, row in dataframe.iterrows():
        content = "\n".join(
            f"{column}: {value}"
            for column, value in row.items()
            if pd.notna(value)
        )

        metadata: dict[str, Any] = {
            "row_index": row_index,
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    logger.info(
        "Created %d documents from %d DataFrame rows.",
        len(documents),
        len(dataframe),
    )

    return documents
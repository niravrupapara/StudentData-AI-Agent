# src/services/query_service.py

from pathlib import Path

import pandas as pd

from src.agent.graph import run_agent_query
from src.data.loader import load_csv_data
from src.data.schema import extract_schema_info
from src.rag.document_builder import dataframe_to_documents
from src.rag.vector_store import (
    create_vector_store,
    index_exists,
    load_vector_store,
    save_vector_store,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def prepare_dataset(
    file_path: str | Path,
    index_name: str = "student_data",
) -> tuple[pd.DataFrame, dict, object]:
    """
    Load the CSV, extract its schema, and prepare the RAG vector store.

    Returns:
        dataframe, schema, vector_store
    """

    logger.info("Preparing dataset: %s", file_path)

    # Load CSV
    dataframe = load_csv_data(file_path)

    # Extract schema
    schema = extract_schema_info(dataframe)

    # Load existing index or create a new one
    if index_exists(index_name):
        logger.info(
            "Existing vector index found. Loading index: %s",
            index_name,
        )

        vector_store = load_vector_store(index_name)

    else:
        logger.info(
            "Vector index not found. Creating index: %s",
            index_name,
        )

        documents = dataframe_to_documents(dataframe)

        vector_store = create_vector_store(documents)

        save_vector_store(
            vector_store=vector_store,
            index_name=index_name,
        )

    logger.info("Dataset preparation completed.")

    return dataframe, schema, vector_store


def process_query(
    graph,
    user_query: str,
) -> str:
    """
    Process a user query through the agent graph.
    """

    logger.info("Processing user query.")

    return run_agent_query(
        graph=graph,
        user_query=user_query,
    )
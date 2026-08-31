import io
from pathlib import Path
from typing import Union

import pandas as pd
from pypdf import PdfReader

from src.utils.logger import get_logger

logger = get_logger(__name__)

FileSource = Union[str, Path, io.BytesIO]


def load_csv(file_source: FileSource) -> pd.DataFrame:
    """Load and validate a CSV file into a Pandas DataFrame."""
    logger.info("Loading CSV data source: %s", file_source)

    try:
        df = pd.read_csv(file_source)

        if df.empty:
            logger.error("Loaded CSV file is empty: %s", file_source)
            raise ValueError("The provided CSV file contains no data.")

        logger.info(
            "CSV loaded successfully | rows=%d | columns=%d",
            len(df),
            len(df.columns),
        )
        return df

    except Exception as e:
        logger.exception("Failed to load CSV file: %s", file_source)
        raise ValueError(f"Error loading CSV file: {e}") from e


def load_excel(file_source: FileSource) -> dict[str, pd.DataFrame]:
    """Load and validate all sheets from an Excel workbook into a dictionary of DataFrames."""
    logger.info("Loading Excel data source: %s", file_source)

    try:
        sheets = pd.read_excel(
            file_source,
            sheet_name=None,
        )

        if not sheets:
            logger.error("Excel file contains no sheets: %s", file_source)
            raise ValueError("The provided Excel file contains no data.")

        logger.info(
            "Excel loaded successfully | sheets_count=%d | sheet_names=%s",
            len(sheets),
            list(sheets.keys()),
        )
        return sheets

    except Exception as e:
        logger.exception("Failed to load Excel file: %s", file_source)
        raise ValueError(f"Error loading Excel file: {e}") from e


def load_pdf(file_source: FileSource) -> str:
    """Extract and validate text content from a PDF file."""
    logger.info("Loading PDF data source: %s", file_source)

    try:
        reader = PdfReader(file_source)

        if not reader.pages:
            logger.error("PDF contains no pages: %s", file_source)
            raise ValueError("The provided PDF contains no pages.")

        pages_text = []
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages_text.append(text)
            logger.debug("PDF page extracted | page=%d | chars=%d", page_idx, len(text))

        content = "\n\n".join(pages_text).strip()

        if not content:
            logger.error("No text could be extracted from PDF: %s", file_source)
            raise ValueError("No extractable text was found in the PDF.")

        logger.info(
            "PDF loaded successfully | total_pages=%d | total_characters=%d",
            len(reader.pages),
            len(content),
        )
        return content

    except Exception as e:
        logger.exception("Failed to load PDF file: %s", file_source)
        raise ValueError(f"Error loading PDF file: {e}") from e


def load_parquet(file_source: FileSource) -> pd.DataFrame:
    """Load and validate a Parquet file into a Pandas DataFrame."""
    logger.info("Loading Parquet data source: %s", file_source)

    try:
        df = pd.read_parquet(file_source)

        if df.empty:
            logger.error("Loaded Parquet file is empty: %s", file_source)
            raise ValueError("The provided Parquet file contains no data.")

        logger.info(
            "Parquet loaded successfully | rows=%d | columns=%d",
            len(df),
            len(df.columns),
        )
        return df

    except Exception as e:
        logger.exception("Failed to load Parquet file: %s", file_source)
        raise ValueError(f"Error loading Parquet file: {e}") from e

def load_text(file_source: FileSource) -> str:
    """Load text file content."""
    if isinstance(file_source, io.BytesIO):
        return file_source.read().decode("utf-8")
    return Path(file_source).read_text(encoding="utf-8")

def load_pdf_pages(file_source: FileSource) -> list[str]:
    """Load PDF into a list of page strings."""
    return [p.extract_text() for p in PdfReader(file_source).pages]

import io
from pathlib import Path
from typing import Union

from pypdf import PdfReader

from src.utils.logger import get_logger

logger = get_logger(__name__)

FileSource = Union[str, Path, io.BytesIO]


def load_pdf(file_source: FileSource) -> str:
    """Extract and validate text from a PDF file."""
    logger.info("Loading PDF data source: %s", file_source)

    try:
        reader = PdfReader(file_source)

        if not reader.pages:
            logger.error("PDF contains no pages.")
            raise ValueError("The provided PDF contains no pages.")

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(text)
            logger.info(
                "PDF page extracted | page=%d | characters=%d",
                page_number,
                len(text),
            )

        content = "\n\n".join(pages).strip()

        if not content:
            logger.error("No text could be extracted from PDF.")
            raise ValueError("No extractable text was found in the PDF.")

        logger.info(
            "PDF loaded successfully | total_pages=%d | total_characters=%d",
            len(reader.pages),
            len(content),
        )
        return content

    except Exception as e:
        logger.exception("Failed to load PDF file.")
        raise ValueError(f"Error loading PDF file: {e}") from e

# src/utils/logger.py

import logging
import os
import sys
from pathlib import Path


# Root directory of the project
BASE_DIR = Path(__file__).resolve().parents[2]

# Ensure logs directory exists
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "app.log"


def setup_logger() -> None:
    """Configure application-wide logging with console and file handlers."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
        force=True,
    )


# Auto-configure once on module import
setup_logger()


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a configured logger for the given module name."""
    return logging.getLogger(name)

import os
from pathlib import Path
import uuid
import matplotlib

# Set non-interactive backend for headless / server execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)

CHARTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "charts"


@tool
def generate_chart(python_code: str) -> str:
    """Execute Python matplotlib/seaborn code to generate and save a chart visualization.
    
    The code should assume standard libraries (plt, sns, pd, np) are available.
    Do NOT call plt.show() inside the code. The tool will save and close the figure automatically.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_id = uuid.uuid4().hex[:8]
    chart_filename = f"chart_{chart_id}.png"
    chart_path = CHARTS_DIR / chart_filename

    logger.info("Executing visualization code | chart_path=%s", chart_path)

    # Clean code formatting if enclosed in backticks
    cleaned_code = python_code.strip()
    if cleaned_code.startswith("```python"):
        cleaned_code = cleaned_code[9:]
    elif cleaned_code.startswith("```"):
        cleaned_code = cleaned_code[3:]
    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code[:-3]
    cleaned_code = cleaned_code.strip()

    # Execution namespace
    exec_globals = {
        "plt": plt,
        "sns": sns,
        "pd": pd,
        "np": np,
        "matplotlib": matplotlib,
    }

    try:
        # Reset any lingering figures
        plt.clf()
        plt.close("all")

        # Execute plotting code
        exec(cleaned_code, exec_globals)

        # Save generated figure
        plt.savefig(
            chart_path,
            bbox_inches="tight",
            dpi=150,
        )
        logger.info("Chart successfully saved | path=%s", chart_path)

        return f"CHART_SAVED: {chart_path.as_posix()}"

    except Exception as e:
        logger.exception("Failed to execute visualization code.")
        return f"ERROR: Failed to generate chart: {e}"

    finally:
        plt.clf()
        plt.close("all")

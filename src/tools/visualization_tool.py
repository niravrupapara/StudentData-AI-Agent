from pathlib import Path
import uuid
import matplotlib

# Non-interactive backend for server/Streamlit execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)

CHARTS_DIR = Path("data/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


@tool
def generate_chart(python_code: str) -> str:
    """Execute Python matplotlib/seaborn code to generate and save a chart visualization to data/charts/.

    The code should assume standard libraries (plt, sns, pd, np) are available.
    All charts will be automatically saved into data/charts/.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_id = uuid.uuid4().hex[:8]
    chart_filename = f"chart_{chart_id}.png"
    chart_path = CHARTS_DIR / chart_filename

    logger.info("Executing chart generation code | target_path=%s", chart_path)

    # Clean markdown formatting if code is enclosed in backticks
    cleaned_code = python_code.strip()
    if cleaned_code.startswith("```python"):
        cleaned_code = cleaned_code[9:]
    elif cleaned_code.startswith("```"):
        cleaned_code = cleaned_code[3:]
    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code[:-3]
    cleaned_code = cleaned_code.strip()

    # Custom savefig wrapper to ensure any savefig call writes to CHARTS_DIR
    def safe_savefig(*args, **kwargs):
        return original_savefig(chart_path, bbox_inches="tight", dpi=150)

    original_savefig = plt.savefig

    exec_globals = {
        "plt": plt,
        "sns": sns,
        "pd": pd,
        "np": np,
        "matplotlib": matplotlib,
    }

    try:
        plt.clf()
        plt.close("all")

        # Temporarily redirect plt.savefig so any script call saves to data/charts/
        plt.savefig = safe_savefig

        # Execute plotting code
        exec(cleaned_code, exec_globals)

        # Restore and save figure if not saved during exec
        plt.savefig = original_savefig
        plt.savefig(chart_path, bbox_inches="tight", dpi=150)

        logger.info("Chart successfully saved | path=%s", chart_path)
        return f"CHART_SAVED: {chart_path.as_posix()}"

    except Exception as e:
        logger.exception("Error executing chart generation.")
        return f"ERROR: Failed to generate chart: {e}"

    finally:
        plt.savefig = original_savefig
        plt.clf()
        plt.close("all")

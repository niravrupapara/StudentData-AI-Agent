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


@tool(response_format="content_and_artifact")
def generate_chart(python_code: str, filename: str = None) -> tuple[str, dict]:
    """Execute Python matplotlib/seaborn code to generate and save a chart visualization to data/charts/.

    The code should assume standard libraries (plt, sns, pd, np) are available.
    All charts will be automatically saved into data/charts/.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_id = uuid.uuid4().hex[:4]
    
    if filename:
        safe_name = filename.strip().replace(" ", "_").replace(".png", "").lower()
        chart_filename = f"{safe_name}_{chart_id}.png"
    else:
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

    original_savefig = plt.savefig
    original_show = plt.show
    original_close = plt.close
    is_saved = False

    def safe_savefig(*args, **kwargs):
        nonlocal is_saved
        is_saved = True
        return original_savefig(chart_path, bbox_inches="tight", dpi=150)

    def safe_show(*args, **kwargs):
        nonlocal is_saved
        if not is_saved:
            original_savefig(chart_path, bbox_inches="tight", dpi=150)
            is_saved = True

    def safe_close(*args, **kwargs):
        nonlocal is_saved
        if not is_saved:
            original_savefig(chart_path, bbox_inches="tight", dpi=150)
            is_saved = True
        return original_close(*args, **kwargs)

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

        # Temporarily redirect matplotlib functions so any call safely writes to chart_path
        plt.savefig = safe_savefig
        plt.show = safe_show
        plt.close = safe_close

        # Execute plotting code
        exec(cleaned_code, exec_globals)

        # If not saved during exec (e.g. script didn't call show/savefig/close), save the active figure
        if not is_saved:
            original_savefig(chart_path, bbox_inches="tight", dpi=150)
            is_saved = True

        logger.info("Chart successfully saved | path=%s", chart_path)
        return (
            "Created chart visualization successfully.",
            {
                "type": "image",
                "path": chart_path.as_posix(),
            },
        )

    except Exception as e:
        logger.exception("Error executing chart generation.")
        return (
            f"ERROR: Failed to generate chart: {e}",
            {
                "type": "error",
                "error": str(e),
            },
        )

    finally:
        plt.savefig = original_savefig
        plt.show = original_show
        plt.close = original_close
        plt.clf()
        original_close("all")

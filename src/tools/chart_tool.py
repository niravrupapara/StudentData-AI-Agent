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
from langchain_experimental.tools import PythonAstREPLTool


from src.utils.logger import get_logger

logger = get_logger(__name__)

CHARTS_DIR = Path("data/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


@tool(response_format="content_and_artifact")
def create_visualization(raw_prompt: str) -> tuple[str, dict]:
    """Create a chart based on the provided data and metadata.
    Pass the exact data values (x and y points) and styling requirements in the prompt.
    """
    logger.info("create_visualization tool called with prompt.")
    
    try:
        from src.agent.chart_agent import generate_chart_code
        # 1. Generate the Python code using the Chart Agent
        python_code, filename = generate_chart_code(raw_prompt)
        
        # 2. Setup execution environment
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

        repl_tool = PythonAstREPLTool()
        
        # We construct a complete script for the REPL to run
        final_code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

plt.clf()
plt.close('all')

# Set a professional look and a constrained figure size
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# --- LLM GENERATED CODE ---
{cleaned_code}
# --------------------------

# Save the plot with tight layout to prevent cutoff labels
plt.tight_layout()
plt.savefig('{chart_path.as_posix()}', bbox_inches='tight', dpi=150)
plt.clf()
plt.close('all')
"""
        try:
            # Execute the code using Langchain's tool
            execution_result = repl_tool.invoke(final_code)
            
            logger.info("Chart successfully saved | path=%s", chart_path)
            return (
                "Created chart visualization successfully.",
                {
                    "type": "image",
                    "path": chart_path.as_posix(),
                },
            )
        except Exception as e:
            logger.exception("Error executing chart generation code.")
            return (
                f"ERROR: Failed to generate chart: {e}",
                {
                    "type": "error",
                    "error": str(e),
                },
            )
            
    except Exception as e:
        logger.exception("Failed to execute generated chart code")
        return f"Error executing chart code: {e}", {"type": "error", "error": str(e)}

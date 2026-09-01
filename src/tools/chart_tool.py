from langchain_core.tools import tool

from src.agent.chart_agent import generate_chart_code
from src.tools.visualization_tool import generate_chart
from src.utils.logger import get_logger

logger = get_logger(__name__)


@tool(response_format="content_and_artifact")
def create_visualization(raw_prompt: str) -> tuple[str, dict]:
    """Create a chart based on the provided data and metadata.
    Pass the exact data values (x and y points) and styling requirements in the prompt.
    """
    logger.info("create_visualization tool called with prompt.")
    
    # 1. Generate the Python code using the Chart Agent
    python_code, filename = generate_chart_code(raw_prompt)
    
    # 2. Execute the code to generate the image
    # Note: generate_chart is decorated with @tool. Since we are in tool-calling context,
    # it's safer to invoke the original underlying python function to get the (result, artifact) tuple.
    # return generate_chart._run(python_code) or similar.
    # However, since generate_chart is imported, we can just use generate_chart.invoke().
    # Let's inspect generate_chart. It is a @tool. 
    # Let's use the underlying function directly to avoid langchain tool nesting issues.
    
    # The @tool decorator in LangChain wraps the original function in `func` property.
    try:
        if hasattr(generate_chart, "func"):
            return generate_chart.func(python_code, filename=filename)
        else:
            # Fallback if it's somehow not wrapped
            return generate_chart(python_code, filename=filename)
    except Exception as e:
        logger.exception("Failed to execute generated chart code")
        return f"Error executing chart code: {e}", {"type": "error", "error": str(e)}

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.llm import get_llm
from src.tools.log_tool import fetch_log_snippets
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT = """You are a Log Analyzer Expert.

Your job is to analyze the provided server or application logs to answer the user's question.

Instructions:
1. Analyze the log data carefully to identify root causes and solutions.
2. Be detailed in your explanation of why an error occurred and how to fix it.
3. If the logs do not contain errors or relevant information, clearly state so.
"""

@tool
def analyze_logs(query: str) -> str:
    """Analyze uploaded log files to find errors, root causes, and solutions.
    Pass the user's query containing the error or issue they are investigating.
    """
    logger.info("analyze_logs tool called in log_agent | query=%s", query)
    
    # Fetch log snippets directly
    logs = fetch_log_snippets.invoke({})
    
    llm = get_llm()
    
    messages = [
        SystemMessage(content=PROMPT),
        HumanMessage(content=f"Log Data:\n\n{logs}\n\nUser Question: {query}")
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.exception("Log Agent failed: %s", e)
        return f"Error analyzing logs: {e}"

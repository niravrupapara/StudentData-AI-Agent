from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

from src.llm import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Cache ──────────────────────────────────────────────────
_DF_REGISTRY: dict[str, pd.DataFrame] = {}   # name → DataFrame
_CACHED_AGENT = None                          # reused across queries


def register_dataframe(name: str, df: pd.DataFrame):
    """Add a DataFrame to the registry. Invalidates the cached agent."""
    global _CACHED_AGENT
    _DF_REGISTRY[name] = df
    _CACHED_AGENT = None  # force rebuild on next query
    logger.info("Registered DataFrame '%s' | shape=%s", name, df.shape)


def get_cached_agent():
    """Return the cached pandas agent, or build one from all registered DFs."""
    global _CACHED_AGENT

    if _CACHED_AGENT is not None:
        return _CACHED_AGENT

    if not _DF_REGISTRY:
        return None

    dfs = list(_DF_REGISTRY.values())
    df_input = dfs[0] if len(dfs) == 1 else dfs

    logger.info("Building pandas agent with %d DataFrame(s)...", len(dfs))
    _CACHED_AGENT = create_pandas_dataframe_agent(
        llm=get_llm(),
        df=df_input,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True,
    )
    return _CACHED_AGENT


def reset_agent():
    """Clear all registered DataFrames and the cached agent."""
    global _CACHED_AGENT
    _DF_REGISTRY.clear()
    _CACHED_AGENT = None
    logger.info("Pandas agent cache cleared.")


@tool
def analyze_data(query: str) -> str:
    """Analyze the uploaded CSV, Excel, or Parquet data using a Pandas agent.

    All structured files are pre-loaded. Just pass your question.

    Args:
        query: Natural-language question about the data.

    Returns:
        Text answer with the analysis result.
    """
    logger.info("Tool analyze_data called | query=%s", query)

    agent = get_cached_agent()
    if agent is None:
        return "No structured data files have been uploaded yet."

    try:
        response = agent.invoke({"input": query})
        return response.get("output", "No response generated.")
    except Exception as e:
        logger.exception("Pandas analysis failed: %s", e)
        return f"Error during data analysis: {e}"

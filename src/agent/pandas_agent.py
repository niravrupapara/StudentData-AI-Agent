from typing import Sequence, Union
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent

from src.llm import get_llm
from src.utils.logger import get_logger


logger = get_logger(__name__)


def create_pandas_agent(df: Union[pd.DataFrame, Sequence[pd.DataFrame]]):
    """Create the ready-made LangChain Pandas DataFrame Agent for single or multiple DataFrames."""
    logger.info("Creating Pandas DataFrame Agent...")
    llm = get_llm()

    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True,
    )


from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHART_SYSTEM_PROMPT = """You are an expert Data Visualization Engineer. 
You will receive data values and chart requirements. 
Write self-contained Python code using matplotlib/seaborn to create the requested chart. 
Embed the provided data directly into the code as lists or numpy arrays. 
Do NOT reference external DataFrames or variables like `df`. 
Only return the python code and a suitable short snake_case filename.
"""

class ChartGenerationResult(BaseModel):
    python_code: str = Field(description="The generated python matplotlib/seaborn code.")
    filename: str = Field(description="A 3-5 word snake_case filename for the chart without the extension, e.g. student_scores_by_class")

def generate_chart_code(raw_prompt: str) -> tuple[str, str]:
    """Generate Python matplotlib code and a filename from a data description prompt."""
    logger.info("Generating chart code for prompt: %s", raw_prompt)
    
    llm = get_llm().with_structured_output(ChartGenerationResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", CHART_SYSTEM_PROMPT),
        ("user", "{raw_prompt}")
    ])
    chain = prompt | llm
    
    response = chain.invoke({"raw_prompt": raw_prompt})
    
    logger.info("Generated code length: %d chars, filename: %s", len(response.python_code), response.filename)
    
    # Strip markdown block if llm included it by mistake
    code = response.python_code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    return code.strip(), response.filename

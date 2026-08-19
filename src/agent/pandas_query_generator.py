from typing import Any

from src.llm.client import llm
from src.utils.logger import get_logger


logger = get_logger(__name__)


PANDAS_QUERY_PROMPT = """
You are a Pandas query generator for a Student Data Agent.

Your job is to convert the user's natural-language question into
ONE valid Pandas expression that operates on a DataFrame named `df`.

Dataset schema:
{schema}

User question:
{user_query}

Rules:
1. Return ONLY the Pandas expression.
2. Do not use markdown.
3. Do not use ```python.
4. The DataFrame variable is always `df`.
5. Use only columns that exist in the provided schema.
6. Do not modify the DataFrame.
7. The expression must return the information needed to answer the question.
8. For filtering, use boolean indexing.
9. For counting, use .count(), .shape[0], or .nunique() as appropriate.
10. For unique values, use .unique().
11. For multiple matching rows, return the relevant columns.
12. Do not write explanations.

Example:

Question:
Who teaches Machine Learning?

Expression:
df[df["Subject"] == "Machine Learning"]["Faculty"].unique()
"""


def generate_pandas_query(
    user_query: str,
    schema: dict[str, Any],
) -> str:
    """
    Generate a Pandas expression from a natural-language question.
    """


    prompt = PANDAS_QUERY_PROMPT.format(
        schema=schema,
        user_query=user_query,
    )

    logger.info("Generating Pandas query for: %s", user_query)

    response = llm.invoke(prompt)

    pandas_query = response.content.strip()

    # Remove accidental markdown fences if the model returns them.
    if pandas_query.startswith("```"):
        lines = pandas_query.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        pandas_query = "\n".join(lines).strip()

    logger.info("Generated Pandas query: %s", pandas_query)

    return pandas_query
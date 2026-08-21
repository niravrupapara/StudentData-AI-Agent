# app.py

import tempfile
from pathlib import Path

import streamlit as st

from src.agent.graph import build_student_agent_graph
from src.data import schema
from src.services.query_service import prepare_dataset, process_query
from src.utils.logger import get_logger


logger = get_logger(__name__)


# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Student Data AI Agent",
    page_icon="🎓",
    layout="wide",
)


# -------------------------------------------------------------------
# Cached Resources
# -------------------------------------------------------------------

@st.cache_resource
def load_agent_graph(
    _dataframe,
    schema,
    _vector_store,
):
    return build_student_agent_graph(
        dataframe=_dataframe,
        schema=schema,
        vector_store=_vector_store,
    )


# -------------------------------------------------------------------
# Application
# -------------------------------------------------------------------

def main():

    st.title("🎓 Student Data AI Agent")
    st.caption(
        "Ask questions about your uploaded student dataset."
    )

    # ---------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------

    with st.sidebar:
        st.header("Upload Data")

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
        )

    # ---------------------------------------------------------------
    # No file uploaded
    # ---------------------------------------------------------------

    if uploaded_file is None:
        st.info("Please upload a CSV file to get started.")
        return

    # ---------------------------------------------------------------
    # Save uploaded file temporarily
    # ---------------------------------------------------------------

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            file_path = Path(temp_file.name)

        logger.info(
            "Uploaded CSV received: %s",
            uploaded_file.name,
        )

        # -----------------------------------------------------------
        # Prepare dataset
        # -----------------------------------------------------------

        with st.spinner("Preparing dataset..."):

            dataframe, schema, vector_store = prepare_dataset(
                file_path=file_path,
            )
            logger.info("Dataset columns: %s", list(dataframe.columns))
            logger.info("Dataset schema: %s", schema)   

        # -----------------------------------------------------------
        # Dataset information
        # -----------------------------------------------------------

        st.success(
            f"Dataset loaded successfully — "
            f"{len(dataframe)} rows, "
            f"{len(dataframe.columns)} columns."
        )

        with st.expander("View Dataset"):

            st.dataframe(
                dataframe,
                use_container_width=True,
            )

        # -----------------------------------------------------------
        # Agent
        # -----------------------------------------------------------

        graph = load_agent_graph(
            _dataframe=dataframe,
            schema=schema,
            _vector_store=vector_store,
        )

        # -----------------------------------------------------------
        # User Query
        # -----------------------------------------------------------

        user_query = st.chat_input(
            "Ask something about the student data..."
        )

        if user_query:

            logger.info(
                "User query received: %s",
                user_query,
            )

            # Display user message
            with st.chat_message("user"):
                st.write(user_query)

            # Generate response
            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    answer = process_query(
                        graph=graph,
                        user_query=user_query,
                    )

                st.write(answer)

    except Exception as error:

        logger.exception(
            "Application error while processing CSV."
        )

        st.error(
            f"Something went wrong: {error}"
        )


if __name__ == "__main__":
    main()
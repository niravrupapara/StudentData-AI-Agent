import uuid
import streamlit as st

from src.agent.graph import build_agent_graph
from src.agent.runner import ask_agent
from src.data_loader import load_csv
from src.utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Data AI Agent",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "graph" not in st.session_state:
    st.session_state.graph = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "df_shape" not in st.session_state:
    st.session_state.df_shape = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        # Only rebuild graph for a new CSV
        if st.session_state.current_file != uploaded_file.name:

            try:
                # Load CSV
                df = load_csv(uploaded_file)

                # Build LangGraph
                st.session_state.graph = build_agent_graph(df)

                # New dataset = new conversation
                st.session_state.thread_id = str(uuid.uuid4())

                st.session_state.current_file = uploaded_file.name

                st.session_state.df_shape = (
                    len(df),
                    len(df.columns)
                )

                logger.info(
                    "CSV loaded | file=%s | thread_id=%s",
                    uploaded_file.name,
                    st.session_state.thread_id
                )

            except Exception as e:

                logger.exception(
                    "Failed to load CSV"
                )

                st.error(
                    f"Error loading CSV: {e}"
                )

        # Dataset information
        if st.session_state.graph is not None:

            rows, cols = st.session_state.df_shape

            st.success("CSV loaded successfully!")

            st.caption(
                f"📊 Rows: {rows} | Columns: {cols}"
            )


# ============================================================
# MAIN UI
# ============================================================

st.title("🤖 Student Data AI Agent")


if st.session_state.graph is None:

    st.info(
        "👈 Upload a CSV file from the sidebar to start chatting."
    )

else:

    # ========================================================
    # LANGGRAPH CONFIG
    # ========================================================

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }


    # ========================================================
    # RESTORE CHAT HISTORY FROM LANGGRAPH
    # ========================================================

    try:

        state_snapshot = (
            st.session_state.graph.get_state(config)
        )

        messages = state_snapshot.values.get(
            "messages",
            []
        )

        for message in messages:

            if message.type == "human":

                with st.chat_message("user"):
                    st.write(message.content)

            elif message.type == "ai":

                with st.chat_message("assistant"):
                    st.write(message.content)

    except Exception as e:

        logger.exception(
            "Failed to restore LangGraph state"
        )

        st.warning(
            f"Unable to restore conversation: {e}"
        )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask a question about your data..."
    )


    if question:

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------

        with st.chat_message("user"):
            st.write(question)


        # ----------------------------------------------------
        # LangGraph execution
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Analyzing your data..."
            ):

                try:

                    answer = ask_agent(
                        graph=st.session_state.graph,
                        question=question,
                        thread_id=st.session_state.thread_id
                    )

                    st.write(answer)

                except Exception as e:

                    logger.exception(
                        "Error executing LangGraph"
                    )

                    st.error(
                        f"Something went wrong: {e}"
                    )
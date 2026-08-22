# app.py

import uuid
import streamlit as st

from src.agent.graph import build_agent_graph
from src.agent.runner import ask_agent
from src.data_loader import load_csv
from src.utils.logger import get_logger


logger = get_logger(__name__)

st.set_page_config(
    page_title="Student Data AI Agent",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# 1. Session State Initialization
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# -----------------------------
# 2. Sidebar (CSV Upload & Stats)
# -----------------------------
with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Rebuild agent only when a new/different CSV is uploaded
        if st.session_state.current_file != uploaded_file.name:
            try:
                df = load_csv(uploaded_file)
                st.session_state.graph = build_agent_graph(df)
                st.session_state.current_file = uploaded_file.name
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.df_shape = (len(df), len(df.columns))
                logger.info("New CSV loaded: %s", uploaded_file.name)
            except Exception as e:
                logger.exception("Failed to load CSV.")
                st.error(f"Error loading CSV: {e}")

        # Show dataset stats
        if st.session_state.graph is not None:
            rows, cols = st.session_state.df_shape
            st.success("CSV loaded successfully!")
            st.caption(f"📊 **Rows:** {rows} | **Columns:** {cols}")

# -----------------------------
# 3. Main Chat Interface
# -----------------------------
st.title("🤖 Student Data AI Agent")

if st.session_state.graph is None:
    st.info("👈 Please upload a CSV file from the sidebar to start chatting.")
else:
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat Input
    if question := st.chat_input("Ask a question about your data..."):
        # 1. Display user query
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # 2. Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                try:
                    answer = ask_agent(
                        st.session_state.graph,
                        question,
                        thread_id=st.session_state.session_id,
                    )
                    st.write(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    logger.exception("Error answering user question.")
                    st.error(f"Something went wrong: {e}")

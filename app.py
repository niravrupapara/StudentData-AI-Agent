from pathlib import Path
import uuid
import streamlit as st

from src.agent.graph import ask_agent, build_agent_graph
from src.agent.pdf_retriever import get_or_create_pdf_retriever
from src.ingestion.csv_loader import load_csv
from src.ingestion.excel_loader import load_excel
from src.ingestion.pdf_loader import load_pdf
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

if "graph" not in st.session_state or st.session_state.graph is None:
    st.session_state.graph = build_agent_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "file_paths" not in st.session_state:
    st.session_state.file_paths = []

if "file_metadata" not in st.session_state:
    st.session_state.file_metadata = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📁 Upload Documents & Data")

    uploaded_files = st.file_uploader(
        "Upload CSV, Excel, or PDF files",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=True,
    )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    if uploaded_files:
        current_file_names = [f.name for f in uploaded_files]
        new_file_paths = []

        for uploaded_file in uploaded_files:
            file_path = upload_dir / uploaded_file.name
            new_file_paths.append(str(file_path.resolve()))

            # Save file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Parse and cache metadata if new file
            if uploaded_file.name not in st.session_state.file_metadata:
                try:
                    ext = file_path.suffix.lower()
                    if ext == ".csv":
                        df = load_csv(file_path)
                        st.session_state.file_metadata[uploaded_file.name] = {
                            "type": "CSV",
                            "info": f"📊 {len(df)} rows | {len(df.columns)} cols",
                        }
                    elif ext in [".xlsx", ".xls"]:
                        sheets = load_excel(file_path)
                        sheet_names = ", ".join(list(sheets.keys()))
                        st.session_state.file_metadata[uploaded_file.name] = {
                            "type": "Excel",
                            "info": f"📈 {len(sheets)} sheet(s): {sheet_names}",
                        }
                    elif ext == ".pdf":
                        # Pre-index vector store in storage upon upload
                        get_or_create_pdf_retriever(file_path)
                        text = load_pdf(file_path)
                        st.session_state.file_metadata[uploaded_file.name] = {
                            "type": "PDF",
                            "info": f"📄 Document indexed ({len(text)} chars)",
                        }

                    logger.info("Processed file upload: %s", uploaded_file.name)

                except Exception as e:
                    logger.exception("Error processing file: %s", uploaded_file.name)
                    st.session_state.file_metadata[uploaded_file.name] = {
                        "type": "Error",
                        "info": f"⚠️ {e}",
                    }

        st.session_state.file_paths = new_file_paths

        # Display file metadata in sidebar
        st.success(f"{len(uploaded_files)} file(s) active!")
        for name, meta in st.session_state.file_metadata.items():
            if name in current_file_names:
                with st.expander(f"{meta['type']}: {name}", expanded=False):
                    st.caption(meta["info"])
    else:
        st.session_state.file_paths = []
        st.session_state.file_metadata = {}


import re

# ============================================================
# VISUALIZATION HELPER
# ============================================================


def render_assistant_message(content: str, tool_messages: list | None = None):
    """Render assistant text response and automatically display any generated chart figures."""
    chart_paths = []

    # 1. Search for chart paths in the assistant's message content
    matches = re.findall(
        r"(?:data[\\/])?charts[\\/]chart_[a-zA-Z0-9_]+\.png",
        content,
    )
    for match in matches:
        p = Path(match)
        if p.exists() and str(p) not in chart_paths:
            chart_paths.append(str(p))

    # 2. Search for chart paths in associated tool messages
    if tool_messages:
        for tm in tool_messages:
            tm_matches = re.findall(
                r"(?:data[\\/])?charts[\\/]chart_[a-zA-Z0-9_]+\.png",
                getattr(tm, "content", ""),
            )
            for match in tm_matches:
                p = Path(match)
                if p.exists() and str(p) not in chart_paths:
                    chart_paths.append(str(p))

    st.write(content)

    # Render any detected chart images
    for chart_path in chart_paths:
        st.image(
            chart_path,
            caption="📊 Generated Visualization",
            use_container_width=True,
        )


# ============================================================
# MAIN UI
# ============================================================

st.title("🤖 Student Data AI Agent")


if not st.session_state.file_paths:

    st.info(
        "👈 Upload one or more CSV, Excel, or PDF files from the sidebar to start chatting."
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
        state_snapshot = st.session_state.graph.get_state(config)
        messages = state_snapshot.values.get("messages", [])

        pending_tools = []
        for message in messages:
            if message.type == "human":
                with st.chat_message("user"):
                    st.write(message.content)
                pending_tools = []

            elif message.type == "tool":
                pending_tools.append(message)

            elif message.type == "ai" and message.content:
                with st.chat_message("assistant"):
                    render_assistant_message(message.content, pending_tools)
                pending_tools = []

    except Exception as e:
        logger.exception("Failed to restore LangGraph state")
        st.warning(f"Unable to restore conversation: {e}")

    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask a question about your uploaded documents or data..."
    )

    if question:

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------
        with st.chat_message("user"):
            st.write(question)

        # ----------------------------------------------------
        # Multi-Agent LangGraph execution
        # ----------------------------------------------------
        with st.chat_message("assistant"):
            with st.spinner("Analyzing across specialized agents..."):
                try:
                    answer = ask_agent(
                        graph=st.session_state.graph,
                        question=question,
                        files=st.session_state.file_paths,
                        thread_id=st.session_state.thread_id,
                    )

                    # Check latest state snapshot for tool messages
                    latest_state = st.session_state.graph.get_state(config)
                    latest_messages = latest_state.values.get("messages", [])
                    latest_tools = [
                        m for m in latest_messages
                        if getattr(m, "type", "") == "tool"
                    ]

                    render_assistant_message(answer, latest_tools)

                except Exception as e:
                    logger.exception("Error executing Multi-Agent LangGraph")
                    st.error(f"Something went wrong: {e}")

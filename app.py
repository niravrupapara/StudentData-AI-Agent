from pathlib import Path
import uuid

import streamlit as st

from src.agent.graph import ask_agent, build_agent_graph
from src.agent.pandas_agent import register_dataframe
from src.ingestion.loaders import load_csv, load_excel, load_parquet, load_pdf_pages, load_text
from src.tools.document_store import register_document
from src.agent.document_agent import invalidate_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Student Data AI Agent",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# SESSION
# ============================================================

if "graph" not in st.session_state:
    st.session_state.graph = build_agent_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "has_files" not in st.session_state:
    st.session_state.has_files = False


# ============================================================
# HELPER - RENDER MESSAGE & CHARTS
# ============================================================

def render_assistant_message(
    content: str,
    artifacts: list[dict] | None = None,
):
    st.markdown(content)

    for artifact in artifacts or []:
        if not isinstance(artifact, dict):
            continue

        if artifact.get("type") != "image":
            continue

        path = Path(artifact.get("path", ""))

        if path.exists():
            st.image(path, width="stretch")
# ============================================================
# SIDEBAR - FILE UPLOAD
# ============================================================

STRUCTURED_EXTS = {".csv", ".xlsx", ".xls", ".parquet"}

with st.sidebar:

    st.header("📁 Upload Files")

    uploaded_files = st.file_uploader(
        "Upload CSV, Excel, Parquet, PDF, or TXT",
        type=["csv", "xlsx", "xls", "parquet", "pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_path = UPLOAD_DIR / uploaded_file.name

            # Save file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            suffix = file_path.suffix.lower()

            if suffix in STRUCTURED_EXTS:
                # Load into DataFrame and register for cached pandas agent
                if suffix == ".csv":
                    df = load_csv(file_path)
                    register_dataframe(uploaded_file.name, df)
                elif suffix in (".xlsx", ".xls"):
                    sheets = load_excel(file_path)
                    for sheet_name, df in sheets.items():
                        register_dataframe(f"{uploaded_file.name}:{sheet_name}", df)
                elif suffix == ".parquet":
                    df = load_parquet(file_path)
                    register_dataframe(uploaded_file.name, df)

            elif suffix == ".pdf":
                register_document(uploaded_file.name, "", file_type="pdf", pages=load_pdf_pages(file_path))
                invalidate_agent()

            elif suffix == ".txt":
                register_document(uploaded_file.name, load_text(file_path), file_type="txt")
                invalidate_agent()

            logger.info("File uploaded: %s", uploaded_file.name)

        st.session_state.has_files = True
        st.success(f"{len(uploaded_files)} file(s) uploaded")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    else:
        st.info("Upload files to start.")


# ============================================================
# MAIN
# ============================================================

st.title("🤖 Student Data AI Agent")

if not st.session_state.has_files:

    st.info("Upload CSV, Excel, Parquet or PDF files from the sidebar.")

else:

    # --------------------------------------------------------
    # Display previous conversation
    # --------------------------------------------------------

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                render_assistant_message(
                    content=msg["content"],
                    artifacts=msg.get("artifacts", []),
                )

    # --------------------------------------------------------
    # Chat
    # --------------------------------------------------------

    question = st.chat_input("Ask about your uploaded files...")

    if question:

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ask_agent(
                        graph=st.session_state.graph,
                        question=question,
                        thread_id=st.session_state.thread_id,
                    )

                    render_assistant_message(
                        content=response["content"],
                        artifacts=response.get("artifacts", []),
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["content"],
                            "artifacts": response.get("artifacts", []),
                        }
                    )

                except Exception as exc:
                    logger.exception("Agent execution failed.")
                    st.error(f"Something went wrong: {exc}")
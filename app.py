from pathlib import Path
import uuid

import streamlit as st

from src.agent.graph import ask_agent, build_agent_graph
from src.tools.pdf_tool import get_or_create_faiss_retriever
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

if "file_paths" not in st.session_state:
    st.session_state.file_paths = []


# ============================================================
# HELPER - RENDER MESSAGE & CHARTS
# ============================================================

def render_assistant_message(content: str, artifacts: list[dict] | None = None):
    """Display assistant response and render any generated chart artifacts."""
    st.write(content)

    if artifacts:
        rendered = set()
        for art in artifacts:
            if isinstance(art, dict) and art.get("type") == "image":
                path_val = art.get("path")
                if path_val:
                    candidates = [
                        Path(path_val),
                        Path(path_val).resolve(),
                        Path("data/charts") / Path(path_val).name,
                    ]
                    for p in candidates:
                        if p.exists() and str(p.resolve()) not in rendered:
                            logger.info("Rendering chart image | path=%s", p)
                            st.image(
                                str(p),
                                caption="📊 Generated Visualization",
                                width="stretch",
                            )
                            rendered.add(str(p.resolve()))
                            break


# ============================================================
# SIDEBAR - FILE UPLOAD
# ============================================================

with st.sidebar:

    st.header("📁 Upload Files")

    uploaded_files = st.file_uploader(
        "Upload CSV, Excel or PDF",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        file_paths = []

        for uploaded_file in uploaded_files:

            file_path = UPLOAD_DIR / uploaded_file.name

            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_paths.append(str(file_path.resolve()))

            # Pre-index PDF with FAISS on upload
            if file_path.suffix.lower() == ".pdf":
                get_or_create_faiss_retriever(file_path)

            logger.info("File uploaded: %s", uploaded_file.name)

        st.session_state.file_paths = file_paths
        st.success(f"{len(uploaded_files)} file(s) uploaded")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    else:
        st.info("Upload files to start.")


# ============================================================
# MAIN
# ============================================================

st.title("🤖 Student Data AI Agent")

if not st.session_state.file_paths:

    st.info("Upload CSV, Excel or PDF files from the sidebar.")

else:

    # --------------------------------------------------------
    # Display previous conversation
    # --------------------------------------------------------

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    try:
        state = st.session_state.graph.get_state(config)
        messages = state.values.get("messages", [])

        pending_artifacts = []
        for message in messages:
            if message.type == "human":
                pending_artifacts = []
                with st.chat_message("user"):
                    st.write(message.content)

            elif message.type == "tool":
                artifact = getattr(message, "artifact", None)
                if isinstance(artifact, dict) and artifact.get("type") == "image":
                    pending_artifacts.append(artifact)

            elif message.type == "ai" and message.content:
                with st.chat_message("assistant"):
                    render_assistant_message(message.content, artifacts=pending_artifacts)
                pending_artifacts = []

    except Exception as exc:
        logger.exception("Failed to load conversation: %s", exc)

    # --------------------------------------------------------
    # Chat
    # --------------------------------------------------------

    question = st.chat_input("Ask about your uploaded files...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ask_agent(
                        graph=st.session_state.graph,
                        question=question,
                        files=st.session_state.file_paths,
                        thread_id=st.session_state.thread_id,
                    )

                    render_assistant_message(
                        content=response["content"],
                        artifacts=response.get("artifacts", []),
                    )

                except Exception as exc:
                    logger.exception("Agent execution failed.")
                    st.error(f"Something went wrong: {exc}")
import re
from pathlib import Path

from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)

UPLOAD_DIR = Path("data/uploads")

@tool
def fetch_log_snippets() -> str:
    """Fetch the contents of uploaded log files. If logs are large, this returns relevant snippets containing errors or exceptions."""
    logger.info("fetch_log_snippets tool invoked")
    
    if not UPLOAD_DIR.exists():
        return "No log files have been uploaded (upload directory not found)."
        
    log_files = list(UPLOAD_DIR.glob("*.log"))
    if not log_files:
        return "No log files (.log) have been uploaded."

    results = []
    for file_path in log_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            
            # Simple heuristic: if file is large, extract only lines with ERROR, Exception, etc.
            if len(lines) > 1000:
                keywords = ["error", "exception", "traceback", "fatal", "warn", "critical", "fail", "timeout", "rejected"]
                relevant_indices = set()
                
                for i, line in enumerate(lines):
                    if any(k in line.lower() for k in keywords):
                        # Add context: 5 lines before, 15 lines after
                        for j in range(max(0, i - 5), min(len(lines), i + 16)):
                            relevant_indices.add(j)
                
                if relevant_indices:
                    relevant_lines = []
                    sorted_indices = sorted(list(relevant_indices))
                    
                    last_idx = -2
                    for idx in sorted_indices:
                        if idx > last_idx + 1 and last_idx != -2:
                            relevant_lines.append("... [SNIP] ...\n")
                        relevant_lines.append(lines[idx])
                        last_idx = idx
                        
                    content = "".join(relevant_lines)
                else:
                    content = "No obvious error keywords found."
                    
                # Truncate if still massive to save context window (roughly 10,000 words ~ 40,000 chars)
                content = content[:40000]
            else:
                content = "".join(lines)
                
            results.append(f"--- File: {file_path.name} ---\n{content}")
        except Exception as e:
            logger.error("Failed to read log file %s: %s", file_path, e)
            results.append(f"--- File: {file_path.name} ---\nError reading file: {e}")

    summary = "\n\n".join(results)
    return summary

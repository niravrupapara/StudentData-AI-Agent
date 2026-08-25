from pathlib import Path


PROJECT_STRUCTURE = {
    "app.py": None,
    ".env": None,
    "requirements.txt": None,

    "config": {
        "__init__.py": None,
        "config.yaml": None,
        "settings.py": None,
    },

    "data": {
        "uploads": {
            ".gitkeep": None,
        },
        "chroma_db": {
            ".gitkeep": None,
        },
        "charts": {
            ".gitkeep": None,
        },
    },

    "logs": {
        ".gitkeep": None,
    },

    "src": {
        "__init__.py": None,
        "llm.py": None,

        "ingestion": {
            "__init__.py": None,
            "csv_loader.py": None,
            "excel_loader.py": None,
            "pdf_loader.py": None,
        },

        "agent": {
            "__init__.py": None,
            "state.py": None,
            "supervisor.py": None,
            "supervisor_tools.py": None,
            "visualization_tool.py": None,
            "pandas_agent.py": None,
            "csv_agent.py": None,
            "excel_agent.py": None,
            "pdf_retriever.py": None,
            "pdf_agent.py": None,
            "graph.py": None,
        },

        "utils": {
            "__init__.py": None,
            "logger.py": None,
        },
    },

    "tests": {
        "__init__.py": None,
        "test_agent.py": None,
    },
}


def create_structure(base_path="."):
    base = Path(base_path)

    def create_items(current_path, structure):
        for name, content in structure.items():
            path = current_path / name

            if isinstance(content, dict):
                path.mkdir(parents=True, exist_ok=True)
                create_items(path, content)
            else:
                if not path.exists():
                    path.touch()
                    print(f"Created: {path}")
                else:
                    print(f"Exists:  {path}")

    create_items(base, PROJECT_STRUCTURE)

    print("\nProject structure created successfully!")


if __name__ == "__main__":
    create_structure()
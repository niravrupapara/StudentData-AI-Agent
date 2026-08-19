from pathlib import Path


PROJECT_STRUCTURE = {
    "config": [
        "__init__.py",
        "settings.py",
    ],
    "src": [
        "__init__.py",
    ],
    "src/agent": [
        "__init__.py",
        "graph.py",
        "state.py",
        "planner.py",
        "response_generator.py",
    ],
    "src/tools": [
        "__init__.py",
        "pandas_tool.py",
        "rag_tool.py",
    ],
    "src/rag": [
        "__init__.py",
        "document_builder.py",
        "embedder.py",
        "vector_store.py",
        "retriever.py",
    ],
    "src/data": [
        "__init__.py",
        "loader.py",
        "schema.py",
    ],
    "src/llm": [
        "__init__.py",
        "client.py",
        "prompts.py",
    ],
    "src/services": [
        "__init__.py",
        "query_service.py",
    ],
    "src/utils": [
        "__init__.py",
        "logger.py",
        "exceptions.py",
    ],
    "data/uploads": [],
    "data/processed": [],
    "storage/indexes": [],
    "logs": [],
}


ROOT_FILES = [
    "app.py",
    "requirements.txt",
    ".env",
    ".gitignore",
]


def create_project_structure():
    root = Path.cwd()

    # Create folders and Python files
    for folder, files in PROJECT_STRUCTURE.items():
        folder_path = root / folder
        folder_path.mkdir(parents=True, exist_ok=True)

        for file_name in files:
            file_path = folder_path / file_name
            file_path.touch(exist_ok=True)

    # Create root-level files
    for file_name in ROOT_FILES:
        file_path = root / file_name
        file_path.touch(exist_ok=True)

    print("Project structure created successfully!")


if __name__ == "__main__":
    create_project_structure()
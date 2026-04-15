import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TARGET_PROJECTS_BASE_PATH = PROJECT_ROOT / "target_github_projects"

CORE_CONCEPT_SEARCH_DIRS = [
    TARGET_PROJECTS_BASE_PATH / "classiq-library/classiq/open_library/functions",
    TARGET_PROJECTS_BASE_PATH / "classiq-library/classiq/qmod/builtins/functions",
    TARGET_PROJECTS_BASE_PATH / "classiq-library/classiq/applications",
]

EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"

SIMILARITY_THRESHOLD = 0.6

TARGET_PROJECTS_BASE_PATH = PROJECT_ROOT / "target_github_projects"

try:
    TARGET_PROJECTS = [
        name
        for name in os.listdir(TARGET_PROJECTS_BASE_PATH)
        if os.path.isdir(TARGET_PROJECTS_BASE_PATH / name)
    ]

except FileNotFoundError:
    print(f"ERROR: The directory '{TARGET_PROJECTS_BASE_PATH}' was not found.")
    print(
        "Please make sure the TARGET_PROJECTS_BASE_PATH in conf/config.py is correct."
    )
    TARGET_PROJECTS = []

SKIP_DIRS_COMMON = [
    "docs",
    "examples",
    "example_notebooks",
    "optimizers",
    "build",
    "dist",
    "__pycache__",
    "venv",
    ".venv",
    "releasenotes",
    "node_modules",
    ".git",
    ".mypy_cache",
    "benchmarks",
    "jupyter_execute",
    "_build",
    "jupyter_notebooks",
    "scripts",
    "tools",
    "dev_tools",
    "binder",
    ".github",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "htmlcov",
    "test_data",
    "utils",
    "tests",
    "test",
]

SKIP_DIRS_FOR_NOTEBOOKS_ONLY = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    "htmlcov",
]

SKIP_FILES_COMMON = ["setup.py", "conftest.py", "__init__.py"]

RESULTS_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

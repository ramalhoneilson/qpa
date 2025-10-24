# justfile

set shell := ["bash", "-c"]

VENV            := ".venv"
REPO_LIST_FILE  := "data/filtered_repo_list.txt"

# It creates the venv, installs ALL dependencies, and clones the repos.
install: discover-and-clone
    @echo "\n>>> Setting up the project environment in '{{VENV}}'..."
    @uv venv --clear --seed {{VENV}}

    @echo " Installing all dependencies from pyproject.toml..."
    @uv sync --python {{VENV}}/bin/python

    @echo " Installing cloned repositories in editable mode..."
    @uv pip install --python {{VENV}}/bin/python -e ./target_github_projects/qiskit
    @uv pip install --python {{VENV}}/bin/python -e ./target_github_projects/pennylane
    @echo ">>> Unified environment setup complete."
    @echo "To activate it manually, run: source {{VENV}}/bin/activate"

# Runs all core concept identification scripts. This is the main analysis command.
# It now depends on `install` to ensure the environment is ready.
identify-concepts: install
    @echo "\n>>> Running all core concept identification tasks..."
    @just identify-qiskit
    @just identify-pennylane
    @just identify-classiq
    @echo "\n All core concept identification tasks are complete."
    @echo "Results are saved in the 'data/' directory."

preprocess-notebooks:
    @echo "\n>>> Preprocessing notebooks (converting to .py and archiving)..."
    @{{VENV}}/bin/python -m src.preprocessing.find_and_copy_notebooks

# Utility to convert notebooks from the archive folder to a separate output folder.
convert-archived-notebooks:
    @echo "\n>>> Converting archived notebooks..."
    @{{VENV}}/bin/python -m src.preprocessing.convert_notebooks

# Downloads the pattern list from the patternatlas.planqk.de website
download_pattern_list:
    @echo "\n>>> Downloading pattern list from patternatlas website..."
    @{{VENV}}/bin/python -m src.preprocessing.download_resources_quantum_patterns

run_main:
    @echo "\n>>> Running main analysis..."
    @{{VENV}}/bin/python -m src.workflows.run_main_analysis

report:
    @echo "\n>>> Generating final report..."
    @{{VENV}}/bin/python -m src.workflows.generate_final_report

# Analyze extended pattern coverage across frameworks and target projects
extended-patterns:
    @echo "\n>>> Analyzing extended pattern coverage..."
    @{{VENV}}/bin/python -m src.utils.analyze_extended_patterns

# Generate PDF files from all Markdown files in docs folder
pdf:
    @echo "\n>>> Generating PDFs from Markdown files..."
    @{{VENV}}/bin/python -m src.utils.generate_pdfs

# Generate experimental data report with complete datasets
experimental-data:
    @echo "\n>>> Generating experimental data report..."
    @{{VENV}}/bin/python -m src.utils.generate_experimental_data_report

# Runs the GitHub search script to find and filter top quantum projects.
search-repos:
    @echo ">>> Running GitHub search to find and filter top quantum projects..."
    @if [ ! -d "{{VENV}}" ]; then just _bootstrap-tools; fi
    @{{VENV}}/bin/python -m src.preprocessing.github_search

# Clones/updates repositories listed in the dynamically generated file.
clone-filtered:
    @echo "\n>>> Cloning/updating repositories from '{{REPO_LIST_FILE}}'..."
    @if [ ! -d "{{VENV}}" ]; then just _bootstrap-tools; fi
    @{{VENV}}/bin/python -m src.preprocessing.clone_repos {{REPO_LIST_FILE}}

# The data acquisition task.
discover-and-clone: search-repos clone-filtered
    @echo "\n>>> All source repositories are ready."


# Identifies and extracts core concepts from the Qiskit source code.
identify-qiskit:
    @echo "\n--- Identifying core concepts in Qiskit ---"
    @{{VENV}}/bin/python -m src.core_concepts.identify_qiskit_core_concepts

# Identifies and extracts core concepts from the PennyLane source code.
identify-pennylane:
    @echo "\n--- Identifying core concepts in PennyLane ---"
    @{{VENV}}/bin/python -m src.core_concepts.identify_pennylane_core_concepts

# Identifies and extracts core concepts from the Classiq source code.
identify-classiq:
    @echo "\n--- Identifying core concepts in Classiq ---"
    @{{VENV}}/bin/python -m src.core_concepts.identify_classiq_core_concepts


# A special recipe to create a  venv just for the data acquisition scripts.
_bootstrap-tools:
    @echo ">>> Bootstrapping minimal tools environment..."
    @uv venv --clear --seed {{VENV}}
    @uv pip install --python {{VENV}}/bin/python PyGithub python-dotenv python-dateutil
    @echo ">>> Bootstrap complete."

# Cleans up ALL generated files and environments.
clean:
    @echo ">>> Cleaning up ALL generated files and environments..."
    @rm -rf {{VENV}} target_github_projects data
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @find . -name "*.ipynb.py" -type f -delete
    @echo ">>> Cleanup complete."

upgrade:
  uv lock --upgrade

# == Testing =================================================================

# Run all tests
test:
    @echo ">>> Running all tests..."
    @{{VENV}}/bin/python -m pytest tests/ -v

# Run unit tests only
test-unit:
    @echo ">>> Running unit tests..."
    @{{VENV}}/bin/python -m pytest tests/ -m unit -v

# Run integration tests only
test-integration:
    @echo ">>> Running integration tests..."
    @{{VENV}}/bin/python -m pytest tests/ -m integration -v

# Run core concepts tests only
test-core-concepts:
    @echo ">>> Running core concepts tests..."
    @{{VENV}}/bin/python -m pytest tests/test_identify_classiq_core_concepts.py -v

# Run tests with coverage
test-coverage:
    @echo ">>> Running tests with coverage..."
    @{{VENV}}/bin/python -m pytest tests/ --cov=src --cov-report=html --cov-report=term -v

# Run tests with coverage and generate detailed report
test-coverage-report:
    @echo ">>> Running tests with detailed coverage report..."
    @{{VENV}}/bin/python -m pytest tests/ --cov=src --cov-report=html --cov-report=xml --cov-report=term --cov-report=json -v
    @echo ">>> Coverage reports generated:"
    @echo "  - HTML: htmlcov/index.html"
    @echo "  - XML: coverage.xml"
    @echo "  - JSON: coverage.json"

# Run coverage analysis on specific module
test-coverage-module module:
    @echo ">>> Running coverage analysis on {{module}}..."
    @{{VENV}}/bin/python -m pytest tests/ --cov={{module}} --cov-report=html --cov-report=term -v

# Generate coverage report without running tests
coverage-report:
    @echo ">>> Generating coverage report from existing data..."
    @{{VENV}}/bin/coverage html
    @{{VENV}}/bin/coverage report
    @echo ">>> HTML report available at: htmlcov/index.html"

# Show coverage summary
coverage-summary:
    @echo ">>> Coverage summary..."
    @{{VENV}}/bin/coverage report --show-missing

# == Code Formatting =========================================================

# Format all Python files with Black
format:
    @echo ">>> Formatting all Python files..."
    uvx ruff format
    @{{VENV}}/bin/black .

# Format specific file or directory
format-file file:
    @echo ">>> Formatting {{file}} with Black..."
    @{{VENV}}/bin/black {{file}}

# Check formatting without making changes
format-check:
    @echo ">>> Checking code formatting with Black..."
    @{{VENV}}/bin/black --check .

# Format and show diff
format-diff:
    @echo ">>> Showing formatting diff with Black..."
    @{{VENV}}/bin/black --diff .

# Lint all Python files with Ruff
lint:
    @echo ">>> Linting all Python files with Ruff..."
    @{{VENV}}/bin/ruff check .

# Lint specific file or directory
lint-file file:
    @echo ">>> Linting {{file}} with Ruff..."
    @{{VENV}}/bin/ruff check {{file}}

# Lint and fix automatically
lint-fix:
    @echo ">>> Linting and fixing all Python files with Ruff..."
    @{{VENV}}/bin/ruff check --fix .

# Lint and fix specific file
lint-fix-file file:
    @echo ">>> Linting and fixing {{file}} with Ruff..."
    @{{VENV}}/bin/ruff check --fix {{file}}

# Run both formatting and linting
format-lint:
    @echo ">>> Running Black formatting and Ruff linting..."
    @just format
    @just lint

# Check formatting and linting without making changes
check-all:
    @echo ">>> Checking formatting and linting..."
    @just format-check
    @just lint

# Format, lint, and run tests
format-lint-test:
    @echo ">>> Running full code quality pipeline..."
    @just format
    @just lint-fix
    @just test

# Run specific test file
test-file file:
    @echo ">>> Running tests in {{file}}..."
    @{{VENV}}/bin/python -m pytest {{file}} -v

# Run tests in parallel
test-parallel:
    @echo ">>> Running tests in parallel..."
    @{{VENV}}/bin/python -m pytest tests/ -n auto -v

# == One-Time Setup ============================================================

default:
  @just --choose

setup:
    @just _setup-{{ os() }}

_setup-macos:
    @echo "Installing uv for macOS..."
    @curl -LsSf https://astral.sh/uv/install.sh | sh

_setup-linux:
    @echo "Installing uv for Linux..."
    @curl -LsSf https://astral.sh/uv/install.sh | sh

_setup-windows:
    @echo "Installing uv for Windows..."
    @powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
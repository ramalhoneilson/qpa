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
    @{{VENV}}/bin/python -m src.preprocessing.extract_notebooks

# Utility to convert notebooks from the archive folder to a separate output folder.
convert-archived-notebooks:
    @echo "\n>>> Converting archived notebooks..."
    @{{VENV}}/bin/python -m src.preprocessing.convert_notebooks

# Downloads the pattern list from the patternatlas.planqk.de website
download_pattern_list:
    @echo "\n>>> Downloading pattern list from patternatlas website..."
    @{{VENV}}/bin/python -m src.data_acquisition.download_patterns

run_main:
    @echo "\n>>> Running main analysis..."
    @{{VENV}}/bin/python -m src.analysis.run_analysis

report:
    @echo "\n>>> Generating final report..."
    @{{VENV}}/bin/python -m src.analysis.generate_report

# Analyze extended pattern coverage across frameworks and target projects
extended-patterns:
    @echo "\n>>> Analyzing extended pattern coverage..."
    @{{VENV}}/bin/python -c "from src.reporting import generate_extended_pattern_analysis; generate_extended_pattern_analysis()"

# Generate PDF files from all Markdown files in docs folder
pdf:
    @echo "\n>>> Generating PDFs from Markdown files..."
    @{{VENV}}/bin/python -c "from src.reporting import generate_pdfs; generate_pdfs()"

# Generate experimental data report with complete datasets
experimental-data:
    @echo "\n>>> Generating experimental data report..."
    @{{VENV}}/bin/python -c "from src.reporting import generate_experimental_data_report; generate_experimental_data_report()"

# Generate base concept report from framework extractions
base-concept-report:
    @echo "\n>>> Generating base concept report..."
    @{{VENV}}/bin/python -c "from src.reporting import generate_base_concept_report; generate_base_concept_report()"

# Generate pattern report from PlanQK Pattern Atlas
pattern-report:
    @echo "\n>>> Generating pattern report..."
    @{{VENV}}/bin/python -c "from src.reporting import generate_pattern_report; generate_pattern_report()"

# Generate all reports at once
all-reports:
    @echo "\n>>> Generating all reports..."
    @{{VENV}}/bin/python -c "from src.reporting import generate_all_reports; generate_all_reports()"

# Consolidate knowledge base from framework data
consolidate-knowledge-base:
    @echo "\n>>> Consolidating knowledge base..."
    @{{VENV}}/bin/python -m src.preprocessing.knowledge_base_consolidator

# Runs the GitHub search script to find and filter top quantum projects.
search-repos:
    @echo ">>> Running GitHub search to find and filter top quantum projects..."
    @if [ ! -d "{{VENV}}" ]; then just _bootstrap-tools; fi
    @{{VENV}}/bin/python -m src.data_acquisition.discover_projects

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

# == Workflow Orchestration ==================================================

# Run the complete workflow using Prefect orchestration
workflow:
    @echo ">>> Starting QPA: Quantum Patterns Analyser Workflow..."
    @{{VENV}}/bin/python run_workflow.py

# Run workflow with Prefect UI (starts local server)
workflow-ui:
    @echo ">>> Starting Prefect UI server..."
    @echo ">>> Open http://localhost:4200 in your browser to monitor the workflow"
    @{{VENV}}/bin/prefect server start --host 0.0.0.0 --port 4200

# Run workflow and deploy to Prefect Cloud (requires account)
workflow-deploy:
    @echo ">>> Deploying workflow to Prefect Cloud..."
    @{{VENV}}/bin/prefect deploy src/workflows/qpa_flow.py:qpa_flow --name qpa-analysis

# Run individual workflow steps for debugging
workflow-step step:
    @echo ">>> Running workflow step: {{step}}"
    @{{VENV}}/bin/python -c "from src.workflows.qpa_flow import {{step}}; {{step}}()"

# Convert Mermaid diagrams to PDF for LaTeX
convert-diagrams:
    @echo ">>> Converting Mermaid diagrams to PDF..."
    @{{VENV}}/bin/python docs/convert_to_pdf.py

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
# Quantum Software Analysis Project

This project provides an automated framework and toolchain for analyzing the source code of popular quantum computing libraries to identify recurring software patterns. It uses `just` as a command runner to orchestrate the entire pipeline, from data collection to final analysis and reporting.

The project features a **dynamic discovery workflow**: it queries the GitHub API to find relevant quantum software projects, clones them, preprocesses their code (including Jupyter Notebooks), and then runs a series of analysis scripts to extract and classify core programming concepts.

## 📊 Experimental Data & Results

### Complete Experimental Datasets

For full reproducibility and transparency, all experimental data is available in the `docs/experimental_data.md` file. This comprehensive dataset includes:

**Framework Concept Extractions:**
- **Table 2**: Complete Classiq Quantum Patterns (`data/classiq_quantum_concepts.csv`)
- **Table 3**: Complete PennyLane Quantum Patterns (`data/pennylane_quantum_concepts.csv`) 
- **Table 4**: Complete Qiskit Quantum Patterns (`data/qiskit_quantum_concepts.csv`)

**Pattern Analysis Results:**
- **Table 5**: Top 10 Most Frequently Matched Quantum Concepts (`data/report/top_matched_concepts.csv`)
- Match Type Analysis (`data/report/match_type_counts.csv`)
- Framework Analysis (`data/report/matches_by_framework.csv`)
- Pattern Frequency Analysis (`data/report/patterns_by_match_count.csv`)

**Pattern Atlas Data:**
- Complete list of quantum patterns from PlanQK Pattern Atlas (`data/quantum_patterns.json`)
- Pattern metadata including names, aliases, intents, and descriptions

### Generate Experimental Data Report

To generate the complete experimental data report:

```bash
just experimental-data
```

This creates `docs/experimental_data.md` with all datasets, properly formatted for academic use with row numbers and complete data (not just summaries).

### Key Findings

The analysis reveals several important patterns in quantum software development:

1. **Framework-Specific Patterns**: Each quantum framework (Qiskit, PennyLane, Classiq) exhibits distinct conceptual patterns
2. **Cross-Framework Similarities**: Common patterns emerge across different frameworks
3. **Pattern Adoption**: How quantum patterns are adopted across different projects
4. **Concept Frequency**: Most frequently used quantum computing concepts



## 🛠 Project Setup & Installation

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.12+**
*   **Just**: A modern command runner. If you don't have it, you can find installation instructions [here](https://github.com/casey/just#installation).
*   **Git**: For cloning the target repositories.
*   **A GitHub Personal Access Token (PAT)**: The discovery script requires a GitHub token to avoid API rate limits. Create a token and save it in a `.env` file in the project root:

    ```text
    # in .env file
    GITHUB_TOKEN="ghp_YourTokenHere"
    ```

## Replication Workflow: Step-by-Step Guide

This guide walks you through the exact sequence of commands to set up the project and replicate the study's results.

### Step 0: Initial Project Setup

This single command prepares the entire project. It will:
1.  Discover and clone the target quantum software repositories from GitHub.
2.  Create a unified virtual environment (`.venv`).
3.  Install all required Python dependencies from `pyproject.toml`.
4.  Install the cloned `qiskit` and `pennylane` repositories in editable mode.

```bash
just install
```
*Note: This command is designed to be fully reproducible and will re-run the setup each time to ensure a clean state. The initial run will take significant time and disk space.*

### Step 1: Download Quantum Pattern Definitions

This step fetches the list of known quantum software patterns from the PlanQK Pattern Atlas, which will be used as a baseline for classification.

```bash
just download_pattern_list
```

### Step 2: Extract Core Concepts from Frameworks

Next, run the scripts that parse the source code of Qiskit, PennyLane, and Classiq to identify their core concepts (e.g., functions and classes).

```bash
just identify-concepts
```

This command generates the following raw data files in the `data/` directory, which you will use in the next step:
*   `data/classiq_quantum_concepts.csv`
*   `data/pennylane_quantum_concepts.csv`
*   `data/qiskit_quantum_concepts.csv`

### Step 3: Manual Concept Classification (Optional)

This is the only manual step in the workflow. The goal is to classify the concepts extracted in the previous step. You have two options:

*   **To replicate our exact results:** You don't need to do anything. The pre-classified files are already provided in the `data/` directory:
    *   `data/enriched_classiq_quantum_patterns.csv`
    *   `data/enriched_pennylane_quantum_patterns.csv`
    *   `data/enriched_qiskit_quantum_patterns.csv`

*   **To perform your own classification:**
    1.  Open the `_quantum_concepts.csv` files generated in Step 2.
    2.  Add your classification data to the rows.
    3.  Save the modified files with the `enriched_` prefix (e.g., `data/enriched_qiskit_quantum_patterns.csv`).

### Step 4.0: Download list of repost from Github
src/preprocessing/github_search.py

### Step 4.1: Preprocess Jupyter Notebooks

This step finds all Jupyter Notebooks (`.ipynb`) within the cloned projects, converts them to Python scripts (`.ipynb.py`) in-place for analysis, and creates an organized archive of the original notebooks.

```bash
just preprocess-notebooks
```

```bash
just convert-archived-notebooks
```

### Step 5: Run the Main Semantic Analysis

With all data prepared, this command runs the main workflow. It uses the `enriched_*.csv` files and the preprocessed source code to search for quantum computing concepts across all target projects.

```bash
just run_main
```
Please ignore the warnings printed in the output as they are just warnings, not errors:

```bash
<unknown>:238: SyntaxWarning: invalid escape sequence '\d'
<unknown>:485: SyntaxWarning: invalid escape sequence '\D'
```

### Step 6: Generate the Final Report

Finally, generate the final report summarizing the findings of the analysis.

```bash
just report
```

This will create the final output files:
- `data/final_pattern_report.txt` - Text summary report
- `docs/final_pattern_report.md` - Markdown report
- `data/report/` - Individual CSV tables for detailed analysis

### Step 7: Generate Experimental Data Report

Generate the complete experimental data report with all datasets:

```bash
just experimental-data
```

This creates `docs/experimental_data.md` with complete datasets for reproducibility.

## 📁 Generated Files & Outputs

### Main Analysis Outputs

**Reports:**
- `docs/final_pattern_report.md` - Main analysis report (Markdown)
- `data/final_pattern_report.txt` - Main analysis report (Text)
- `docs/experimental_data.md` - Complete experimental datasets

**CSV Data Tables:**
- `data/classiq_quantum_concepts.csv` - Classiq framework concepts
- `data/pennylane_quantum_concepts.csv` - PennyLane framework concepts  
- `data/qiskit_quantum_concepts.csv` - Qiskit framework concepts
- `data/quantum_patterns.json` - Pattern Atlas data

**Analysis Results:**
- `data/report/top_matched_concepts.csv` - Most frequently matched concepts
- `data/report/match_type_counts.csv` - Match type distribution
- `data/report/matches_by_framework.csv` - Framework analysis
- `data/report/patterns_by_match_count.csv` - Pattern frequency
- `data/report/source_pattern_analysis.csv` - Source pattern analysis
- `data/report/adoption_pattern_analysis.csv` - Pattern adoption analysis

### Intermediate Files

**Preprocessed Code:**
- `notebooks/` - Converted Jupyter notebooks (`.ipynb.py`)
- `converted_notebooks/` - Archive of original notebooks
- `target_github_projects/` - Cloned quantum software repositories

**Configuration:**
- `.venv/` - Python virtual environment
- `uv.lock` - Dependency lock file
- `.env` - Environment variables (GitHub token)

## Command Reference

You can always run `just` to see an interactive list of available commands.

### Main Workflow Commands

*   `install`: Sets up the entire project, including cloning, environment creation, and dependency installation.
*   `identify-concepts`: Runs the core concept extraction for Qiskit, PennyLane, and Classiq.
*   `run_main`: Executes the primary semantic analysis workflow.
*   `report`: Generates the final summary report.

### Individual Data & Preprocessing Steps

*   `download_pattern_list`: Fetches pattern definitions from the PlanQK Pattern Atlas.
*   `discover-and-clone`: Runs only the GitHub search and cloning steps.
*   `preprocess-notebooks`: Converts `.ipynb` files to `.py` and creates an archive.
*   `convert-archived-notebooks`: A separate utility to convert notebooks from the archive folder.

### Utility Commands

*   `clean`: Removes **ALL** generated artifacts: the virtual environment, all cloned code, and the `data`, `notebooks`, and `converted_notebooks` directories. Use this for a complete reset.
*   `upgrade`: Updates the `uv.lock` file based on `pyproject.toml`. Run this after changing dependencies.
*   `setup`: A one-time command to install the `uv` package manager.

### Testing & Development Commands

*   `test`: Run all tests with coverage
*   `test-coverage`: Run tests with detailed coverage report
*   `test-file <file>`: Run tests for a specific file
*   `format`: Format all Python files with Black
*   `lint`: Run linting with Ruff
*   `format-lint-test`: Run formatting, linting, and testing in sequence

## 🧪 Testing & Quality Assurance

This project includes comprehensive testing and quality assurance:

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing  
- **Coverage Reports**: Detailed code coverage analysis
- **Automated Testing**: GitHub Actions CI/CD pipeline

### Code Quality
- **Black**: Automatic code formatting
- **Ruff**: Fast Python linting and fixing
- **Type Hints**: Full type annotation support
- **Documentation**: Comprehensive docstrings and README

### Development Workflow
```bash
# Run tests
just test

# Format and lint code
just format-lint-test

# Generate experimental data
just experimental-data
```

## 📚 Documentation

- **Main README**: This file - project overview and setup
- **Experimental Data**: `docs/experimental_data.md` - Complete datasets
- **Coverage Report**: `docs/COVERAGE.md` - Testing documentation
- **Formatting Guide**: `docs/FORMATTING.md` - Code style guidelines
- **Refactoring Summary**: `docs/refactoring_summary.md` - Architecture documentation

## 🏗 Project Architecture

### Core Components

**Data Processing:**
- `src/core_concepts/` - Framework concept extraction
- `src/preprocessing/` - Data preparation and notebook conversion
- `src/workflows/` - Main analysis workflows

**Utilities:**
- `src/utils/` - Report generation and data export
- `src/conf/` - Configuration management

**Testing:**
- `tests/` - Comprehensive test suite
- `pytest.ini` - Test configuration
- `.coveragerc` - Coverage settings

### Key Features

- **Modular Design**: Single Responsibility Principle with separated concerns
- **Comprehensive Testing**: 50+ test cases with full coverage
- **Automated Workflows**: Complete pipeline automation with `just`
- **Quality Assurance**: Code formatting, linting, and testing
- **Reproducible Results**: Complete experimental data export
- **Academic Standards**: Proper citations and documentation

### File Structure

```
quantum_patterns/
├── src/                          # Source code
│   ├── core_concepts/           # Concept extraction
│   ├── preprocessing/           # Data preparation  
│   ├── workflows/              # Analysis workflows
│   └── utils/                  # Utilities
├── tests/                       # Test suite
├── docs/                       # Documentation
├── data/                       # Generated data
├── notebooks/                  # Converted notebooks
├── converted_notebooks/        # Notebook archive
├── target_github_projects/     # Cloned repositories
├── justfile                    # Command automation
├── pyproject.toml             # Project configuration
└── README.md                   # This file
```

## 🤝 Contributing

This project follows best practices for scientific software:

1. **Reproducible Research**: All data and code are version controlled
2. **Comprehensive Testing**: Full test coverage with automated CI/CD
3. **Code Quality**: Automated formatting and linting
4. **Documentation**: Complete documentation for all components
5. **Modular Architecture**: Clean separation of concerns

For development, use the provided commands:
```bash
just format-lint-test  # Format, lint, and test
just test-coverage     # Run with coverage
just experimental-data # Generate data report
```

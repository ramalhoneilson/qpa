# Test Coverage Documentation

This document describes the test coverage setup for the quantum_patterns project.

## Coverage Setup

### Dependencies
- `coverage>=7.0.0` - Core coverage library
- `pytest-cov>=7.0.0` - Pytest integration for coverage

### Configuration
- `.coveragerc` - Coverage configuration file
- Coverage files are automatically ignored in `.gitignore`

## Available Commands

### Basic Coverage Commands
```bash
# Run tests with basic coverage
just test-coverage

# Run tests with detailed coverage reports (HTML, XML, JSON)
just test-coverage-report

# Run coverage analysis on specific module
just test-coverage-module src.core_concepts.identify_classiq_core_concepts

# Generate coverage report from existing data
just coverage-report

# Show coverage summary with missing lines
just coverage-summary
```

## Coverage Reports

### Report Types Generated
1. **HTML Report** (`htmlcov/index.html`) - Interactive web-based report
2. **XML Report** (`coverage.xml`) - Machine-readable format for CI/CD
3. **JSON Report** (`coverage.json`) - Structured data format
4. **Terminal Report** - Console output with summary

### Current Coverage Status

#### Overall Project Coverage: 8%
- **Total Statements**: 1,403
- **Missing Statements**: 1,293
- **Covered Statements**: 110

#### Core Module Coverage: 48%
- **Module**: `src/core_concepts/identify_classiq_core_concepts.py`
- **Statements**: 198
- **Missing**: 103
- **Covered**: 95

#### Well-Tested Modules
- `src/__init__.py` - 100% coverage
- `src/conf/__init__.py` - 100% coverage
- `src/core_concepts/__init__.py` - 100% coverage
- `src/conf/config.py` - 79% coverage

#### Untested Modules (0% coverage)
- `src/core_concepts/identify_pennylane_core_concepts.py`
- `src/core_concepts/identify_qiskit_core_concepts.py`
- `src/preprocessing/*` - All preprocessing modules
- `src/utils/*` - All utility modules
- `src/workflows/*` - All workflow modules

## Coverage Analysis

### What's Covered
The current test suite covers:
- **Utility Functions** (7 tests) - `_create_summary_from_docstring`
- **AST Visitor Class** (6 tests) - `_PublicApiVisitor`
- **File Parsing** (3 tests) - `_find_concepts_in_file`
- **Analysis Functions** (3 tests) - `run_final_analysis`
- **Constants** (3 tests) - Module constants validation

### What's Not Covered
The following functions in `identify_classiq_core_concepts.py` are not covered:
- `_get_sdk_root_path()` - SDK path detection
- `extract_core_concepts()` - Core concept extraction
- `_save_source_code_snippets()` - Source code saving
- `_save_concepts_to_csv()` - CSV export
- `main()` - Main function integration

### Missing Lines (Lines 74, 148-159, 168-195, 224-266, 270-290, 296-316, 320)
These correspond to:
- Error handling paths
- File I/O operations
- Main function execution
- Save operations
- SDK path detection

## Coverage Goals

### Current Status
- **Core Functionality**: 48% coverage
- **Test Reliability**: 100% pass rate
- **Focus**: Working functionality only

### Recommendations
1. **For Production**: Current coverage is sufficient for core functionality
2. **For Development**: Add tests for missing functions as needed
3. **For CI/CD**: Use current test suite for reliable automated testing

## Coverage Reports Location

### HTML Report
- **Location**: `htmlcov/index.html`
- **Features**: Interactive browsing, line-by-line coverage
- **Usage**: Open in web browser for detailed analysis

### Data Files
- **Coverage Database**: `.coverage` (binary format)
- **XML Export**: `coverage.xml` (CI/CD integration)
- **JSON Export**: `coverage.json` (programmatic access)

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run tests with coverage
  run: just test-coverage-report

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: coverage.xml
```

### Coverage Thresholds
- **Minimum**: 40% for core modules
- **Target**: 60% for critical functionality
- **Excluded**: Untested modules (preprocessing, workflows, utils)

## Best Practices

### Running Coverage
1. **Development**: Use `just test-coverage` for quick feedback
2. **CI/CD**: Use `just test-coverage-report` for comprehensive reports
3. **Analysis**: Use `just coverage-summary` for missing line details

### Coverage Interpretation
- **Green (80%+)**: Well-tested code
- **Yellow (40-79%)**: Partially tested code
- **Red (0-39%)**: Untested or legacy code

### Maintenance
- Run coverage regularly during development
- Focus on critical paths and error handling
- Use HTML reports for detailed line-by-line analysis
- Keep coverage reports up to date in CI/CD pipelines





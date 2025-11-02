"""
Test configuration and shared fixtures for the QPA (Quantum Patterns Analyser) test suite.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, mock_open, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_source_code():
    """Sample Python source code for testing."""
    return '''
def public_function():
    """This is a public function with a docstring."""
    return "Hello, World!"

def private_function():
    """This is a private function."""
    return "Private"

def function_without_docstring():
    return "No docstring"

def function_with_boilerplate():
    """[Qmod Classiq-library function] This is a real docstring."""
    return "With boilerplate"
'''


@pytest.fixture
def sample_concepts_data():
    """Sample concepts data for testing."""
    return [
        {
            "name": "/classiq/test.module.function1",
            "summary": "Test function 1",
            "docstring": "This is test function 1",
            "source_code": "def function1():\n    pass",
        },
        {
            "name": "/classiq/test.module.function2",
            "summary": "Test function 2",
            "docstring": "This is test function 2",
            "source_code": "def function2():\n    pass",
        },
    ]


@pytest.fixture
def mock_classiq_module():
    """Mock classiq module for testing."""
    mock_module = Mock()
    mock_module.__path__ = ["/test/classiq"]
    return mock_module


@pytest.fixture
def mock_target_module():
    """Mock target module with __all__ attribute."""
    mock_module = Mock()
    mock_module.__all__ = ["function1", "function2", "function3"]
    return mock_module


@pytest.fixture
def sample_ast_tree():
    """Sample AST tree for testing."""
    import ast

    source = '''
def test_function():
    """Test function docstring."""
    pass
'''
    return ast.parse(source)


@pytest.fixture
def mock_file_path():
    """Mock file path for testing."""
    return Path("/test/sdk/open_library/functions/test_module.py")


@pytest.fixture
def mock_sdk_root():
    """Mock SDK root path for testing."""
    return Path("/test/sdk")


@pytest.fixture
def public_api_names():
    """Sample public API names for testing."""
    return {"test_function", "public_function"}


@pytest.fixture
def mock_logging():
    """Mock logging for testing."""
    with patch("src.core_concepts.identify_classiq_core_concepts.logging") as mock_log:
        yield mock_log


@pytest.fixture
def mock_json_dump():
    """Mock json.dump for testing."""
    with patch("json.dump") as mock_dump:
        yield mock_dump


@pytest.fixture
def mock_csv_writer():
    """Mock CSV writer for testing."""
    with patch("csv.DictWriter") as mock_writer:
        mock_writer_instance = Mock()
        mock_writer.return_value = mock_writer_instance
        yield mock_writer_instance


@pytest.fixture
def mock_open_file():
    """Mock open function for file operations."""
    with patch("builtins.open", mock_open()) as mock_file:
        yield mock_file


@pytest.fixture
def sample_docstrings():
    """Sample docstrings for testing summary creation."""
    return {
        "empty": "",
        "single_sentence": "This is a single sentence.",
        "two_sentences": "First sentence. Second sentence. Third sentence.",
        "multiline": """First paragraph with multiple lines.
        
        Second paragraph.""",
        "with_whitespace": "   First sentence.   Second sentence.   ",
        "no_periods": "This is a sentence without periods",
        "boilerplate": "[Qmod Classiq-library function] This is a real docstring.",
        "only_boilerplate": "[Qmod Classiq-library function]",
    }


@pytest.fixture
def expected_summaries():
    """Expected summaries for docstring tests."""
    return {
        "empty": "",
        "single_sentence": "This is a single sentence.",
        "two_sentences": "First sentence. Second sentence.",
        "multiline": "First paragraph with multiple lines.",
        "with_whitespace": "First sentence. Second sentence.",
        "no_periods": "This is a sentence without periods",
        "boilerplate": "This is a real docstring.",
        "only_boilerplate": "",
    }


# Test data for integration tests
@pytest.fixture
def integration_test_data():
    """Test data for integration tests."""
    return {
        "source_code": '''
def amplitude_amplification():
    """Implements amplitude amplification algorithm."""
    pass

def grover_search():
    """Implements Grover's search algorithm."""
    pass
''',
        "expected_concepts": ["amplitude_amplification", "grover_search"],
        "public_api": {"amplitude_amplification", "grover_search"},
    }


# Mock configurations for different test scenarios
@pytest.fixture
def mock_config_success():
    """Mock configuration for successful execution."""
    return {
        "sdk_path": Path("/test/classiq"),
        "modules": ["classiq.open_library.functions"],
        "public_apis": {"function1", "function2"},
        "concepts_found": [
            {"name": "test.function1", "summary": "Test 1"},
            {"name": "test.function2", "summary": "Test 2"},
        ],
    }


@pytest.fixture
def mock_config_failure():
    """Mock configuration for failure scenarios."""
    return {
        "sdk_path": None,
        "modules": ["classiq.open_library.functions"],
        "public_apis": set(),
        "concepts_found": [],
    }


# Utility functions for tests
def create_temp_python_file(directory: Path, filename: str, content: str) -> Path:
    """Create a temporary Python file for testing."""
    file_path = directory / filename
    file_path.write_text(content)
    return file_path


def create_temp_directory_structure(base_path: Path, structure: dict[str, Any]) -> None:
    """Create a temporary directory structure for testing."""
    for name, content in structure.items():
        path = base_path / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_temp_directory_structure(path, content)
        else:
            path.write_text(content)


# Test markers for different test categories
pytestmark = [pytest.mark.unit, pytest.mark.core_concepts]

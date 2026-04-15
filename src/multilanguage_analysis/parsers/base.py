"""Abstract interface for language-specific code parsers."""

from abc import ABC, abstractmethod
from pathlib import Path


class CodeParser(ABC):
    """Extracts call sites and comments from a source file.

    Implementations return data in a language-agnostic format so that
    run_analysis_multilang.py can treat all languages identically.
    """

    @abstractmethod
    def extract_calls(self, source: str) -> list[tuple[str, int]]:
        """Return a list of (call_name, line_number) tuples found in *source*.

        *call_name* is the bare function/method name without qualifiers.
        *line_number* is 1-based.
        """

    @abstractmethod
    def extract_comments(self, source: str) -> str:
        """Return all comment text in *source* as a single space-joined string."""

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """File extensions this parser handles, e.g. ['.cpp', '.cc', '.cxx']."""

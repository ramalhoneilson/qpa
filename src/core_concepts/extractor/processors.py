# src/core_concepts/extractor/processors.py

import re
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """
    Abstract base class for processing docstrings from different sources.

    This class defines the interface that all source-specific processors must
    implement, ensuring that the core `ConceptExtractor` can work with any of
    them in a uniform way.
    """

    @abstractmethod
    def clean_docstring(self, docstring: str) -> str:
        """
        Takes a raw docstring and returns a cleaned version.

        Cleaning may involve removing boilerplate text, stripping whitespace, etc.

        Args:
            docstring: The raw docstring extracted from the AST.

        Returns:
            The processed, clean docstring.
        """
        pass

    @abstractmethod
    def create_summary(self, cleaned_docstring: str) -> str:
        """
        Creates a brief summary from a cleaned docstring.

        Typically, this involves extracting the first sentence or paragraph.

        Args:
            cleaned_docstring: The docstring after processing by clean_docstring().

        Returns:
            A short, one-line or two-sentence summary.
        """
        pass


class ClassiqProcessor(BaseProcessor):
    """Processor for Classiq-style docstrings."""

    # A list of boilerplate strings commonly found in Classiq docstrings.
    BOILERPLATE = [
        "[Qmod Classiq-library function]",
        "[Qmod core-library function]",
    ]

    def clean_docstring(self, docstring: str) -> str:
        """
        Removes known boilerplate strings and extra whitespace from Classiq docstrings.
        """
        if not docstring:
            return ""

        cleaned = docstring
        for bp_string in self.BOILERPLATE:
            # This regex handles optional whitespace and a final period.
            pattern = re.compile(r"\s*" + re.escape(bp_string) + r"\.?\s*")
            # Replace with a single space to avoid merging words, then strip later.
            cleaned = pattern.sub(" ", cleaned)

        return cleaned.strip()

    def create_summary(self, cleaned_docstring: str) -> str:
        """
        Extracts the first one or two sentences to create a summary for Classiq concepts.
        """
        if not cleaned_docstring:
            return ""

        first_paragraph = cleaned_docstring.strip().split("\n\n")[0]
        text_block = " ".join(first_paragraph.split())  # Normalize whitespace

        # Split into sentences based on periods.
        parts = text_block.split(".")
        sentences = [s.strip() for s in parts if s.strip()]

        if len(sentences) >= 2:
            return f"{sentences[0]}. {sentences[1]}."
        elif sentences:
            return f"{sentences[0]}."
        else:
            # Fallback for docstrings without periods.
            return first_paragraph


class PennylaneProcessor(BaseProcessor):
    """Processor for PennyLane-style docstrings."""

    def clean_docstring(self, docstring: str) -> str:
        """
        Performs basic cleaning for PennyLane docstrings (mostly stripping whitespace).
        """
        return docstring.strip() if docstring else ""

    def create_summary(self, cleaned_docstring: str) -> str:
        """
        Creates a summary from the first paragraph of a PennyLane docstring.
        """
        if not cleaned_docstring:
            return ""

        # PennyLane's summary logic is simpler: take the first paragraph and
        # replace any internal newlines with spaces.
        summary = cleaned_docstring.split("\n\n")[0].strip().replace("\n", " ")
        return summary


class QiskitProcessor(BaseProcessor):
    """Processor for Qiskit-style docstrings, preserving original cleaning logic."""

    def clean_docstring(self, docstring: str) -> str:
        """Removes code blocks, circuit symbols, and math directives from docstrings."""
        if not docstring:
            return ""
        # Pattern for code blocks and parsed literals
        pattern = r"(.. (code-block|parsed-literal):: text\n\n)(^\s+.*$\n?)+"
        cleaned = re.sub(pattern, "", docstring, flags=re.MULTILINE)

        # Pattern for circuit symbols
        circuit_symbol_pattern = r".*Circuit symbol:.*(?:\n\s*.. code-block:: text)?\n\n(^\s*.*[┌┐└┘├┤│─].*$\n?)+"
        cleaned = re.sub(circuit_symbol_pattern, "", cleaned, flags=re.MULTILINE)

        # Pattern for math directives
        math_pattern = r"(.. math::\n\n)(^\s+.*$\n?)+"
        cleaned = re.sub(math_pattern, "", cleaned, flags=re.MULTILINE)

        return cleaned.strip()

    def create_summary(self, cleaned_docstring: str) -> str:
        """Creates a summary from the first paragraph of a Qiskit docstring."""
        if not cleaned_docstring:
            return ""
        return cleaned_docstring.split("\n\n")[0].strip().replace("\n", " ")

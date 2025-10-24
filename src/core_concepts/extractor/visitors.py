# src/core_concepts/extractor/visitors.py

import ast
import logging
from pathlib import Path
from typing import Any, Dict, Set

# Import the processor base class for type hinting
from .processors import BaseProcessor

# A type alias for clarity
Concept = Dict[str, Any]


class BaseConceptVisitor(ast.NodeVisitor):
    """
    An abstract base class for AST visitors that find code concepts.

    This class provides the common initialization logic for all specific
    visitors. It is responsible for storing the source text, file paths,
    and the injected docstring processor. Subclasses must implement their
    own `visit_*` methods (e.g., `visit_FunctionDef`, `visit_ClassDef`)
    to define the actual extraction logic.
    """

    def __init__(
        self,
        source_text: str,
        file_path: Path,
        sdk_root: Path,
        processor: BaseProcessor,
        **kwargs: Any,
    ):
        """
        Initializes the base visitor.

        Args:
            source_text: The full source code of the file being parsed.
            file_path: The path to the file being parsed.
            sdk_root: The root path of the SDK/library being scanned.
            processor: An instance of a docstring processor.
            **kwargs: Catches any extra arguments for subclasses.
        """
        self.found_concepts: Dict[str, Concept] = {}
        self.source_text = source_text
        self.file_path = file_path
        self.sdk_root = sdk_root
        self.processor = processor


class ClassiqFunctionVisitor(BaseConceptVisitor):
    """

    An AST visitor that finds public API functions from Classiq.

    It identifies functions by checking if their names are present in a
    provided set of public API names (`__all__`). It only considers
    functions that have a meaningful docstring after cleaning.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # This visitor requires a set of public API names to be passed in.
        self.public_api_names: Set[str] = kwargs.get("public_api_names", set())
        if not self.public_api_names:
            logging.warning(
                "ClassiqFunctionVisitor initialized with no public_api_names."
            )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """This method is called by the AST walker for every function definition."""
        if node.name in self.public_api_names:
            raw_docstring = ast.get_docstring(node)
            if not raw_docstring:
                logging.debug(f"Skipping '{node.name}': No docstring.")
                self.generic_visit(node)
                return

            cleaned_docstring = self.processor.clean_docstring(raw_docstring)
            if not cleaned_docstring:
                logging.warning(
                    f"Skipping '{node.name}' in {self.file_path.name}: docstring contained only boilerplate."
                )
                self.generic_visit(node)
                return

            self._add_concept(node, cleaned_docstring, "/classiq")

        # Continue traversing the tree to find nested functions if any.
        self.generic_visit(node)

    def _add_concept(self, node: ast.FunctionDef, cleaned_docstring: str, prefix: str):
        """Helper to create and store a concept dictionary."""
        summary = self.processor.create_summary(cleaned_docstring)

        relative_path = self.file_path.relative_to(self.sdk_root)
        module_path = ".".join(list(relative_path.parts)[:-1] + [relative_path.stem])
        full_name = f"{prefix}/{module_path}.{node.name}"

        if full_name not in self.found_concepts:
            self.found_concepts[full_name] = {
                "name": full_name,
                "summary": summary,
                "docstring": cleaned_docstring,
                "source_code": ast.get_source_segment(self.source_text, node),
            }
            logging.debug(f"Found concept: {node.name}")


class PennylaneClassVisitor(BaseConceptVisitor):
    """
    An AST visitor that finds classes with docstrings in PennyLane.

    It identifies concepts by looking for any class definition (`ClassDef`)
    that has a docstring associated with it.
    """

    def visit_ClassDef(self, node: ast.ClassDef):
        """This method is called by the AST walker for every class definition."""
        raw_docstring = ast.get_docstring(node)

        if raw_docstring:
            cleaned_docstring = self.processor.clean_docstring(raw_docstring)
            if cleaned_docstring:
                self._add_concept(node, cleaned_docstring, "/pennylane")

        # Continue traversing the tree.
        self.generic_visit(node)

    def _add_concept(self, node: ast.ClassDef, cleaned_docstring: str, prefix: str):
        """Helper to create and store a concept dictionary."""
        summary = self.processor.create_summary(cleaned_docstring)

        relative_path = self.file_path.relative_to(self.sdk_root)
        module_path = ".".join(list(relative_path.parts)[:-1] + [relative_path.stem])
        full_name = f"{prefix}/{module_path}.{node.name}"

        if full_name not in self.found_concepts:
            self.found_concepts[full_name] = {
                "name": full_name,
                "summary": summary,
                "docstring": cleaned_docstring,
                "source_code": ast.get_source_segment(self.source_text, node),
            }
            logging.debug(f"Found concept: {node.name}")


TARGET_BASE_CLASSES = ["QuantumCircuit", "Gate"]


class QiskitVisitor(BaseConceptVisitor):
    """
    An AST visitor that finds both classes and top-level functions in Qiskit,
    preserving the detailed logic from the original script.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.context_stack: list[ast.AST] = []

    def _get_module_path_str(self) -> str:
        """Helper to generate the fully qualified module path string."""
        relative_path = self.file_path.relative_to(self.sdk_root)
        module_path_parts = list(relative_path.parts)
        module_path_parts[-1] = relative_path.stem
        return ".".join(module_path_parts)

    def _visit_context_node(self, node: ast.AST):
        """Helper to manage the context stack during traversal."""
        self.context_stack.append(node)
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visits class definitions, extracting detailed metadata."""
        raw_docstring = ast.get_docstring(node)
        if raw_docstring:
            docstring = self.processor.clean_docstring(raw_docstring)
            if docstring:
                module_path_str = self._get_module_path_str()
                full_concept_name = f"/qiskit/{module_path_str}.{node.name}"

                if full_concept_name not in self.found_concepts:
                    base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
                    self.found_concepts[full_concept_name] = {
                        "name": full_concept_name,
                        "summary": self.processor.create_summary(docstring),
                        "docstring": docstring,
                        "source_code": ast.get_source_segment(self.source_text, node),
                        "type": "Class",
                        "is_target_subclass": any(
                            base in TARGET_BASE_CLASSES for base in base_names
                        ),
                        "base_classes": base_names,
                    }
        self._visit_context_node(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visits function definitions, ignoring class methods."""
        # Check context_stack to ensure this is a top-level function, not a method
        if not any(isinstance(p, ast.ClassDef) for p in self.context_stack):
            raw_docstring = ast.get_docstring(node)
            # Additional original filtering logic
            if (
                raw_docstring
                and not node.name.startswith("_")
                and not node.name.startswith("get_")
            ):
                docstring = self.processor.clean_docstring(raw_docstring)
                if docstring:
                    module_path_str = self._get_module_path_str()
                    full_concept_name = f"/qiskit/{module_path_str}.{node.name}"

                    if full_concept_name not in self.found_concepts:
                        self.found_concepts[full_concept_name] = {
                            "name": full_concept_name,
                            "summary": self.processor.create_summary(docstring),
                            "docstring": docstring,
                            "source_code": ast.get_source_segment(
                                self.source_text, node
                            ),
                            "type": "Function",
                        }
        self._visit_context_node(node)

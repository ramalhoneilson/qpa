"""
Tests for src/core_concepts/extractor/visitors.py

This module tests the AST visitor classes for extracting quantum computing
concepts from different frameworks (Classiq, PennyLane, Qiskit).
"""

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core_concepts.extractor.visitors import (
    BaseConceptVisitor,
    ClassiqFunctionVisitor,
    PennylaneClassVisitor,
    QiskitVisitor,
    TARGET_BASE_CLASSES,
    Concept,
)


class TestBaseConceptVisitor:
    """Test the BaseConceptVisitor abstract base class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_processor = MagicMock()
        self.source_text = "def test_function(): pass"
        self.file_path = Path("/test/file.py")
        self.sdk_root = Path("/test")
        
        # Create a concrete implementation for testing
        class TestVisitor(BaseConceptVisitor):
            def visit_FunctionDef(self, node):
                pass
        
        self.visitor = TestVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )

    def test_initialization(self):
        """Test BaseConceptVisitor initialization."""
        assert self.visitor.found_concepts == {}
        assert self.visitor.source_text == self.source_text
        assert self.visitor.file_path == self.file_path
        assert self.visitor.sdk_root == self.sdk_root
        assert self.visitor.processor == self.mock_processor

    def test_initialization_with_kwargs(self):
        """Test initialization with additional kwargs."""
        extra_kwargs = {"extra_param": "value"}
        
        class TestVisitor(BaseConceptVisitor):
            def visit_FunctionDef(self, node):
                pass
        
        visitor = TestVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor,
            **extra_kwargs
        )
        
        # Should not raise an error and initialize normally
        assert visitor.found_concepts == {}

    def test_concept_type_alias(self):
        """Test that Concept type alias is properly defined."""
        # This is more of a type checking test
        concept: Concept = {
            "name": "test_concept",
            "summary": "test summary",
            "docstring": "test docstring",
            "source_code": "def test(): pass"
        }
        assert isinstance(concept, dict)
        assert "name" in concept
        assert "summary" in concept
        assert "docstring" in concept
        assert "source_code" in concept


class TestClassiqFunctionVisitor:
    """Test the ClassiqFunctionVisitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_processor = MagicMock()
        self.source_text = """
def public_function():
    \"\"\"This is a public function.\"\"\"
    pass

def private_function():
    \"\"\"This is a private function.\"\"\"
    pass
"""
        self.file_path = Path("/classiq/sdk/module.py")
        self.sdk_root = Path("/classiq/sdk")
        self.public_api_names = {"public_function"}

    def test_initialization_with_public_api_names(self):
        """Test initialization with public API names."""
        visitor = ClassiqFunctionVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor,
            public_api_names=self.public_api_names
        )
        
        assert visitor.public_api_names == self.public_api_names
        assert visitor.found_concepts == {}

    def test_initialization_without_public_api_names(self):
        """Test initialization without public API names."""
        with patch('logging.warning') as mock_warning:
            visitor = ClassiqFunctionVisitor(
                source_text=self.source_text,
                file_path=self.file_path,
                sdk_root=self.sdk_root,
                processor=self.mock_processor
            )
            
            assert visitor.public_api_names == set()
            mock_warning.assert_called_once_with(
                "ClassiqFunctionVisitor initialized with no public_api_names."
            )

    def test_visit_function_def_public_function(self):
        """Test visiting a public function with docstring."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = ClassiqFunctionVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor,
            public_api_names=self.public_api_names
        )
        
        # Parse the source and visit the function
        tree = ast.parse(self.source_text)
        function_node = tree.body[0]  # public_function
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="def public_function(): pass"), \
             patch('logging.debug') as mock_debug:
            
            visitor.visit_FunctionDef(function_node)
            
            # Should have found the concept
            assert len(visitor.found_concepts) == 1
            concept = list(visitor.found_concepts.values())[0]
            assert concept["name"] == "/classiq/module.public_function"
            assert concept["summary"] == "Summary"
            assert concept["docstring"] == "Cleaned docstring"
            assert concept["source_code"] == "def public_function(): pass"
            
            # Check processor calls
            self.mock_processor.clean_docstring.assert_called_once_with("Original docstring")
            self.mock_processor.create_summary.assert_called_once_with("Cleaned docstring")
            mock_debug.assert_called_once_with("Found concept: public_function")

    def test_visit_function_def_private_function(self):
        """Test visiting a private function (not in public API)."""
        visitor = ClassiqFunctionVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor,
            public_api_names=self.public_api_names
        )
        
        # Parse the source and visit the function
        tree = ast.parse(self.source_text)
        function_node = tree.body[1]  # private_function
        
        visitor.visit_FunctionDef(function_node)
        
        # Should not have found any concepts
        assert len(visitor.found_concepts) == 0

    def test_visit_function_def_no_docstring(self):
        """Test visiting a function without docstring."""
        visitor = ClassiqFunctionVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor,
            public_api_names=self.public_api_names
        )
        
        tree = ast.parse(self.source_text)
        function_node = tree.body[0]  # public_function
        
        with patch('ast.get_docstring', return_value=None), \
             patch('logging.debug') as mock_debug:
            
            visitor.visit_FunctionDef(function_node)
            
            # Should not have found any concepts
            assert len(visitor.found_concepts) == 0
            mock_debug.assert_called_once_with("Skipping 'public_function': No docstring.")

    def test_visit_function_def_empty_docstring_after_cleaning(self):
        """Test visiting a function with docstring that becomes empty after cleaning."""
        self.mock_processor.clean_docstring.return_value = ""
        
        visitor = ClassiqFunctionVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor,
            public_api_names=self.public_api_names
        )
        
        tree = ast.parse(self.source_text)
        function_node = tree.body[0]  # public_function
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('logging.warning') as mock_warning:
            
            visitor.visit_FunctionDef(function_node)
            
            # Should not have found any concepts
            assert len(visitor.found_concepts) == 0
            mock_warning.assert_called_once_with(
                "Skipping 'public_function' in module.py: docstring contained only boilerplate."
            )

    def test_add_concept_duplicate_prevention(self):
        """Test that duplicate concepts are not added."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = ClassiqFunctionVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor,
            public_api_names=self.public_api_names
        )
        
        tree = ast.parse(self.source_text)
        function_node = tree.body[0]  # public_function
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="def public_function(): pass"), \
             patch('logging.debug'):
            
            # Visit the same function twice
            visitor.visit_FunctionDef(function_node)
            visitor.visit_FunctionDef(function_node)
            
            # Should only have one concept
            assert len(visitor.found_concepts) == 1

    def test_module_path_generation(self):
        """Test that module paths are generated correctly."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = ClassiqFunctionVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor,
            public_api_names=self.public_api_names
        )
        
        tree = ast.parse(self.source_text)
        function_node = tree.body[0]  # public_function
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="def public_function(): pass"), \
             patch('logging.debug'):
            
            visitor.visit_FunctionDef(function_node)
            
            concept = list(visitor.found_concepts.values())[0]
            assert concept["name"] == "/classiq/module.public_function"


class TestPennylaneClassVisitor:
    """Test the PennylaneClassVisitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_processor = MagicMock()
        self.source_text = """
class PublicClass:
    \"\"\"This is a public class.\"\"\"
    pass

class PrivateClass:
    \"\"\"This is a private class.\"\"\"
    pass

class NoDocstringClass:
    pass
"""
        self.file_path = Path("/pennylane/sdk/module.py")
        self.sdk_root = Path("/pennylane/sdk")

    def test_visit_class_def_with_docstring(self):
        """Test visiting a class with docstring."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = PennylaneClassVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[0]  # PublicClass
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="class PublicClass: pass"), \
             patch('logging.debug') as mock_debug:
            
            visitor.visit_ClassDef(class_node)
            
            # Should have found the concept
            assert len(visitor.found_concepts) == 1
            concept = list(visitor.found_concepts.values())[0]
            assert concept["name"] == "/pennylane/module.PublicClass"
            assert concept["summary"] == "Summary"
            assert concept["docstring"] == "Cleaned docstring"
            assert concept["source_code"] == "class PublicClass: pass"
            
            # Check processor calls
            self.mock_processor.clean_docstring.assert_called_once_with("Original docstring")
            self.mock_processor.create_summary.assert_called_once_with("Cleaned docstring")
            mock_debug.assert_called_once_with("Found concept: PublicClass")

    def test_visit_class_def_no_docstring(self):
        """Test visiting a class without docstring."""
        visitor = PennylaneClassVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[2]  # NoDocstringClass
        
        with patch('ast.get_docstring', return_value=None):
            visitor.visit_ClassDef(class_node)
            
            # Should not have found any concepts
            assert len(visitor.found_concepts) == 0

    def test_visit_class_def_empty_docstring_after_cleaning(self):
        """Test visiting a class with docstring that becomes empty after cleaning."""
        self.mock_processor.clean_docstring.return_value = ""
        
        visitor = PennylaneClassVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[0]  # PublicClass
        
        with patch('ast.get_docstring', return_value="Original docstring"):
            visitor.visit_ClassDef(class_node)
            
            # Should not have found any concepts
            assert len(visitor.found_concepts) == 0

    def test_visit_class_def_duplicate_prevention(self):
        """Test that duplicate concepts are not added."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = PennylaneClassVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[0]  # PublicClass
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="class PublicClass: pass"), \
             patch('logging.debug'):
            
            # Visit the same class twice
            visitor.visit_ClassDef(class_node)
            visitor.visit_ClassDef(class_node)
            
            # Should only have one concept
            assert len(visitor.found_concepts) == 1

    def test_module_path_generation(self):
        """Test that module paths are generated correctly."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = PennylaneClassVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[0]  # PublicClass
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="class PublicClass: pass"), \
             patch('logging.debug'):
            
            visitor.visit_ClassDef(class_node)
            
            concept = list(visitor.found_concepts.values())[0]
            assert concept["name"] == "/pennylane/module.PublicClass"


class TestQiskitVisitor:
    """Test the QiskitVisitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_processor = MagicMock()
        self.source_text = """
class QuantumCircuit:
    \"\"\"A quantum circuit class.\"\"\"
    pass

class Gate:
    \"\"\"A gate class.\"\"\"
    pass

class CustomGate(Gate):
    \"\"\"A custom gate class.\"\"\"
    pass

def public_function():
    \"\"\"A public function.\"\"\"
    pass

def _private_function():
    \"\"\"A private function.\"\"\"
    pass

def get_something():
    \"\"\"A getter function.\"\"\"
    pass

class SomeClass:
    def method(self):
        \"\"\"A method.\"\"\"
        pass
"""
        self.file_path = Path("/qiskit/sdk/module.py")
        self.sdk_root = Path("/qiskit/sdk")

    def test_initialization(self):
        """Test QiskitVisitor initialization."""
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        assert visitor.found_concepts == {}
        assert visitor.context_stack == []

    def test_get_module_path_str(self):
        """Test module path string generation."""
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        module_path = visitor._get_module_path_str()
        assert module_path == "module"

    def test_visit_context_node(self):
        """Test context node management."""
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[0]  # QuantumCircuit
        
        # Mock the generic_visit to avoid actual traversal
        with patch.object(visitor, 'generic_visit'):
            visitor._visit_context_node(class_node)
            
            # Context stack should be empty after visiting
            assert visitor.context_stack == []

    def test_visit_class_def_with_docstring(self):
        """Test visiting a class with docstring."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[0]  # QuantumCircuit
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="class QuantumCircuit: pass"), \
             patch.object(visitor, '_visit_context_node') as mock_visit_context:
            
            visitor.visit_ClassDef(class_node)
            
            # Should have found the concept
            assert len(visitor.found_concepts) == 1
            concept = list(visitor.found_concepts.values())[0]
            assert concept["name"] == "/qiskit/module.QuantumCircuit"
            assert concept["summary"] == "Summary"
            assert concept["docstring"] == "Cleaned docstring"
            assert concept["source_code"] == "class QuantumCircuit: pass"
            assert concept["type"] == "Class"
            assert concept["is_target_subclass"] is False
            assert concept["base_classes"] == []
            
            # Should have called context node visit
            mock_visit_context.assert_called_once_with(class_node)

    def test_visit_class_def_with_base_classes(self):
        """Test visiting a class with base classes."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[2]  # CustomGate(Gate)
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="class CustomGate(Gate): pass"), \
             patch.object(visitor, '_visit_context_node'):
            
            visitor.visit_ClassDef(class_node)
            
            # Should have found the concept
            assert len(visitor.found_concepts) == 1
            concept = list(visitor.found_concepts.values())[0]
            assert concept["name"] == "/qiskit/module.CustomGate"
            assert concept["is_target_subclass"] is True  # Gate is in TARGET_BASE_CLASSES
            assert concept["base_classes"] == ["Gate"]

    def test_visit_class_def_no_docstring(self):
        """Test visiting a class without docstring."""
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[0]  # QuantumCircuit
        
        with patch('ast.get_docstring', return_value=None), \
             patch.object(visitor, '_visit_context_node') as mock_visit_context:
            
            visitor.visit_ClassDef(class_node)
            
            # Should not have found any concepts
            assert len(visitor.found_concepts) == 0
            # Should still visit context node
            mock_visit_context.assert_called_once_with(class_node)

    def test_visit_function_def_public_function(self):
        """Test visiting a public function."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        function_node = tree.body[3]  # public_function
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="def public_function(): pass"), \
             patch.object(visitor, '_visit_context_node') as mock_visit_context:
            
            visitor.visit_FunctionDef(function_node)
            
            # Should have found the concept
            assert len(visitor.found_concepts) == 1
            concept = list(visitor.found_concepts.values())[0]
            assert concept["name"] == "/qiskit/module.public_function"
            assert concept["summary"] == "Summary"
            assert concept["docstring"] == "Cleaned docstring"
            assert concept["source_code"] == "def public_function(): pass"
            assert concept["type"] == "Function"
            
            # Should have called context node visit
            mock_visit_context.assert_called_once_with(function_node)

    def test_visit_function_def_private_function(self):
        """Test visiting a private function (starts with underscore)."""
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        function_node = tree.body[4]  # _private_function
        
        with patch.object(visitor, '_visit_context_node') as mock_visit_context:
            visitor.visit_FunctionDef(function_node)
            
            # Should not have found any concepts
            assert len(visitor.found_concepts) == 0
            # Should still visit context node
            mock_visit_context.assert_called_once_with(function_node)

    def test_visit_function_def_getter_function(self):
        """Test visiting a getter function (starts with get_)."""
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        function_node = tree.body[5]  # get_something
        
        with patch.object(visitor, '_visit_context_node') as mock_visit_context:
            visitor.visit_FunctionDef(function_node)
            
            # Should not have found any concepts
            assert len(visitor.found_concepts) == 0
            # Should still visit context node
            mock_visit_context.assert_called_once_with(function_node)

    def test_visit_function_def_method_in_class(self):
        """Test visiting a method inside a class (should be ignored)."""
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[6]  # SomeClass
        method_node = class_node.body[0]  # method
        
        # Simulate being inside a class by adding it to context stack
        visitor.context_stack = [class_node]
        
        with patch.object(visitor, '_visit_context_node') as mock_visit_context:
            visitor.visit_FunctionDef(method_node)
            
            # Should not have found any concepts (it's a method, not a top-level function)
            assert len(visitor.found_concepts) == 0
            # Should still visit context node
            mock_visit_context.assert_called_once_with(method_node)

    def test_visit_function_def_no_docstring(self):
        """Test visiting a function without docstring."""
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        function_node = tree.body[3]  # public_function
        
        with patch('ast.get_docstring', return_value=None), \
             patch.object(visitor, '_visit_context_node') as mock_visit_context:
            
            visitor.visit_FunctionDef(function_node)
            
            # Should not have found any concepts
            assert len(visitor.found_concepts) == 0
            # Should still visit context node
            mock_visit_context.assert_called_once_with(function_node)

    def test_visit_function_def_empty_docstring_after_cleaning(self):
        """Test visiting a function with docstring that becomes empty after cleaning."""
        self.mock_processor.clean_docstring.return_value = ""
        
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        function_node = tree.body[3]  # public_function
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch.object(visitor, '_visit_context_node') as mock_visit_context:
            
            visitor.visit_FunctionDef(function_node)
            
            # Should not have found any concepts
            assert len(visitor.found_concepts) == 0
            # Should still visit context node
            mock_visit_context.assert_called_once_with(function_node)

    def test_duplicate_prevention(self):
        """Test that duplicate concepts are not added."""
        self.mock_processor.clean_docstring.return_value = "Cleaned docstring"
        self.mock_processor.create_summary.return_value = "Summary"
        
        visitor = QiskitVisitor(
            source_text=self.source_text,
            file_path=self.file_path,
            sdk_root=self.sdk_root,
            processor=self.mock_processor
        )
        
        tree = ast.parse(self.source_text)
        class_node = tree.body[0]  # QuantumCircuit
        
        with patch('ast.get_docstring', return_value="Original docstring"), \
             patch('ast.get_source_segment', return_value="class QuantumCircuit: pass"), \
             patch.object(visitor, '_visit_context_node'):
            
            # Visit the same class twice
            visitor.visit_ClassDef(class_node)
            visitor.visit_ClassDef(class_node)
            
            # Should only have one concept
            assert len(visitor.found_concepts) == 1


class TestTargetBaseClasses:
    """Test the TARGET_BASE_CLASSES constant."""

    def test_target_base_classes_constant(self):
        """Test that TARGET_BASE_CLASSES is defined correctly."""
        assert TARGET_BASE_CLASSES == ["QuantumCircuit", "Gate"]
        assert isinstance(TARGET_BASE_CLASSES, list)
        assert len(TARGET_BASE_CLASSES) == 2


class TestIntegration:
    """Integration tests for the visitor classes."""

    def test_classiq_function_visitor_integration(self):
        """Test ClassiqFunctionVisitor with real AST parsing."""
        mock_processor = MagicMock()
        mock_processor.clean_docstring.return_value = "Cleaned docstring"
        mock_processor.create_summary.return_value = "Summary"
        
        source_text = """
def public_function():
    \"\"\"This is a public function.\"\"\"
    pass
"""
        
        visitor = ClassiqFunctionVisitor(
            source_text=source_text,
            file_path=Path("/classiq/sdk/module.py"),
            sdk_root=Path("/classiq/sdk"),
            processor=mock_processor,
            public_api_names={"public_function"}
        )
        
        # Parse and visit the AST
        tree = ast.parse(source_text)
        visitor.visit(tree)
        
        # Should have found the concept
        assert len(visitor.found_concepts) == 1
        concept = list(visitor.found_concepts.values())[0]
        assert concept["name"] == "/classiq/module.public_function"

    def test_pennylane_class_visitor_integration(self):
        """Test PennylaneClassVisitor with real AST parsing."""
        mock_processor = MagicMock()
        mock_processor.clean_docstring.return_value = "Cleaned docstring"
        mock_processor.create_summary.return_value = "Summary"
        
        source_text = """
class PublicClass:
    \"\"\"This is a public class.\"\"\"
    pass
"""
        
        visitor = PennylaneClassVisitor(
            source_text=source_text,
            file_path=Path("/pennylane/sdk/module.py"),
            sdk_root=Path("/pennylane/sdk"),
            processor=mock_processor
        )
        
        # Parse and visit the AST
        tree = ast.parse(source_text)
        visitor.visit(tree)
        
        # Should have found the concept
        assert len(visitor.found_concepts) == 1
        concept = list(visitor.found_concepts.values())[0]
        assert concept["name"] == "/pennylane/module.PublicClass"

    def test_qiskit_visitor_integration(self):
        """Test QiskitVisitor with real AST parsing."""
        mock_processor = MagicMock()
        mock_processor.clean_docstring.return_value = "Cleaned docstring"
        mock_processor.create_summary.return_value = "Summary"
        
        source_text = """
class QuantumCircuit:
    \"\"\"A quantum circuit class.\"\"\"
    pass

def public_function():
    \"\"\"A public function.\"\"\"
    pass
"""
        
        visitor = QiskitVisitor(
            source_text=source_text,
            file_path=Path("/qiskit/sdk/module.py"),
            sdk_root=Path("/qiskit/sdk"),
            processor=mock_processor
        )
        
        # Parse and visit the AST
        tree = ast.parse(source_text)
        visitor.visit(tree)
        
        # Should have found both concepts
        assert len(visitor.found_concepts) == 2
        concept_names = [concept["name"] for concept in visitor.found_concepts.values()]
        assert "/qiskit/module.QuantumCircuit" in concept_names
        assert "/qiskit/module.public_function" in concept_names

    def test_edge_case_handling(self):
        """Test edge cases in visitor functionality."""
        mock_processor = MagicMock()
        mock_processor.clean_docstring.return_value = "Cleaned docstring"
        mock_processor.create_summary.return_value = "Summary"
        
        # Test with empty source
        visitor = QiskitVisitor(
            source_text="",
            file_path=Path("/qiskit/sdk/module.py"),
            sdk_root=Path("/qiskit/sdk"),
            processor=mock_processor
        )
        
        tree = ast.parse("")
        visitor.visit(tree)
        
        # Should not have found any concepts
        assert len(visitor.found_concepts) == 0

    def test_complex_nested_structure(self):
        """Test visitors with complex nested structures."""
        mock_processor = MagicMock()
        mock_processor.clean_docstring.return_value = "Cleaned docstring"
        mock_processor.create_summary.return_value = "Summary"
        
        source_text = """
class OuterClass:
    \"\"\"An outer class.\"\"\"
    
    class InnerClass:
        \"\"\"An inner class.\"\"\"
        pass
    
    def method(self):
        \"\"\"A method.\"\"\"
        pass

def top_level_function():
    \"\"\"A top-level function.\"\"\"
    pass
"""
        
        visitor = QiskitVisitor(
            source_text=source_text,
            file_path=Path("/qiskit/sdk/module.py"),
            sdk_root=Path("/qiskit/sdk"),
            processor=mock_processor
        )
        
        # Parse and visit the AST
        tree = ast.parse(source_text)
        visitor.visit(tree)
        
        # Should have found the outer class, inner class, and top-level function
        # Method should be ignored (it's inside a class)
        assert len(visitor.found_concepts) == 3
        concept_names = [concept["name"] for concept in visitor.found_concepts.values()]
        assert "/qiskit/module.OuterClass" in concept_names
        assert "/qiskit/module.InnerClass" in concept_names
        assert "/qiskit/module.top_level_function" in concept_names

"""AST-based call graph extractor.

For each Python file, extracts one DefinedConcept per class or top-level
function.  Each DefinedConcept records:
  - The names of everything the body *calls*  (function/method call sites)
  - The names of all *base classes*           (inheritance)
  - The docstring                             (for fallback pattern resolution)

This is the opposite of the call-site analyser in run_analysis.py.
That script asks: "what quantum concepts does this file *use*?"
This module asks: "what does each defined concept *depend on*?"
"""
import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DefinedConcept:
    name: str
    file_path: str
    docstring: str
    bases: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)


class _CallCollector(ast.NodeVisitor):
    """Collects every called name inside a subtree."""

    def __init__(self):
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            self.calls.append(func.id)
        elif isinstance(func, ast.Attribute):
            # e.g. self.qft() → "qft";  QuantumCircuit.compose() → "compose"
            self.calls.append(func.attr)
        self.generic_visit(node)


def _base_names(bases: list[ast.expr]) -> list[str]:
    names = []
    for base in bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def extract_defined_concepts(file_path: Path) -> list[DefinedConcept]:
    """Parse *file_path* and return one DefinedConcept per class/top-level function."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []

    concepts: list[DefinedConcept] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            collector = _CallCollector()
            collector.visit(node)
            concepts.append(
                DefinedConcept(
                    name=node.name,
                    file_path=str(file_path),
                    docstring=ast.get_docstring(node) or "",
                    bases=_base_names(node.bases),
                    calls=collector.calls,
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            collector = _CallCollector()
            collector.visit(node)
            concepts.append(
                DefinedConcept(
                    name=node.name,
                    file_path=str(file_path),
                    docstring=ast.get_docstring(node) or "",
                    bases=[],
                    calls=collector.calls,
                )
            )

    return concepts

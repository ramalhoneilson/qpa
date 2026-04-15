"""C++ source parser backed by tree-sitter-cpp.

Extracts function/method call names and comment text from .cpp/.cc/.cxx/.h/.hpp
files.  Handles all C++ call forms:

  simple()                     → "simple"
  obj.method()                 → "method"
  ns::func()                   → "func"
  obj->method()                → "method"
  tmpl<Arg>()                  → "tmpl"

Block comments (/* */) and line comments (//) are both captured.
"""

from tree_sitter import Language, Node, Parser

import tree_sitter_cpp as tscpp

from .base import CodeParser

_CPP_LANGUAGE = Language(tscpp.language())
_PARSER = Parser(_CPP_LANGUAGE)


def _walk(node: Node):
    """Depth-first traversal yielding every node in the tree."""
    yield node
    for child in node.children:
        yield from _walk(child)


class CppParser(CodeParser):
    @property
    def file_extensions(self) -> list[str]:
        return [".cpp", ".cc", ".cxx", ".h", ".hpp"]

    def extract_calls(self, source: str) -> list[tuple[str, int]]:
        """Return (call_name, line_number) pairs from all call sites."""
        try:
            tree = _PARSER.parse(bytes(source, "utf-8", errors="replace"))
        except Exception:
            return []

        results: list[tuple[str, int]] = []
        for node in _walk(tree.root_node):
            if node.type != "call_expression":
                continue

            func = node.child_by_field_name("function")
            if func is None:
                continue

            name: str | None = None

            if func.type == "identifier":
                # simple_call(...)
                name = func.text.decode("utf-8", errors="replace")

            elif func.type == "field_expression":
                # obj.method(...) or obj->method(...)
                field = func.child_by_field_name("field")
                if field:
                    name = field.text.decode("utf-8", errors="replace")

            elif func.type == "scoped_identifier":
                # ns::func(...)
                scope_name = func.child_by_field_name("name")
                if scope_name:
                    name = scope_name.text.decode("utf-8", errors="replace")

            elif func.type in ("template_function", "template_method"):
                # func<Arg>(...) — extract the base name
                name_node = func.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode("utf-8", errors="replace")

            if name:
                line = node.start_point[0] + 1  # convert to 1-based
                results.append((name, line))

        return results

    def extract_comments(self, source: str) -> str:
        """Return all comment text joined into a single string."""
        try:
            tree = _PARSER.parse(bytes(source, "utf-8", errors="replace"))
        except Exception:
            return ""

        parts: list[str] = []
        for node in _walk(tree.root_node):
            if node.type == "comment":
                raw = node.text.decode("utf-8", errors="replace")
                # Strip comment delimiters
                text = raw.lstrip("/").lstrip("*").rstrip("*").rstrip("/").strip()
                if text:
                    parts.append(text)

        return " ".join(parts)

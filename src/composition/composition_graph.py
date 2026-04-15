"""Build a quantum composition graph from defined concepts.

An edge  A --[calls]--> B  means: a concept resolved to pattern A has a call
site for something resolved to pattern B inside its body.

An edge  A --[inherits]--> B  means: a concept resolved to pattern A extends
a class resolved to pattern B.

Self-edges (A → A) are suppressed — they represent internal details of a
pattern, not inter-pattern composition.

Duplicate edges from the same (from_concept, to_concept) pair are collapsed;
the relationship type is preserved.
"""
from dataclasses import dataclass
from pathlib import Path

from .call_graph import DefinedConcept
from .pattern_resolver import PatternResolver


@dataclass
class ResolvedConcept:
    name: str
    file_path: str
    pattern: str
    resolution_type: str   # "name" | "docstring"
    score: float


@dataclass
class CompositionEdge:
    from_name: str
    from_pattern: str
    to_name: str
    to_pattern: str
    relationship: str      # "calls" | "inherits"
    from_file: str


def build_composition_graph(
    concepts: list[DefinedConcept],
    resolver: PatternResolver,
) -> tuple[list[ResolvedConcept], list[CompositionEdge]]:
    """Resolve concepts to patterns and emit composition edges.

    Returns
    -------
    resolved : list[ResolvedConcept]
        Every defined concept that could be mapped to a quantum pattern.
    edges : list[CompositionEdge]
        Directed composition edges between patterns (self-edges excluded).
    """
    # ── Step 1: resolve every defined concept to a pattern ──────────────────
    # Try name first; fall back to docstring for library-style classes whose
    # name might not be in the KB but whose docstring describes the pattern.
    resolved: dict[str, ResolvedConcept] = {}  # concept.name → ResolvedConcept

    for c in concepts:
        pattern, score = resolver.resolve_name(c.name)
        res_type = "name"
        if pattern is None and c.docstring:
            pattern, score = resolver.resolve_docstring(c.docstring)
            res_type = "docstring"
        if pattern is not None:
            resolved[c.name] = ResolvedConcept(
                name=c.name,
                file_path=c.file_path,
                pattern=pattern,
                resolution_type=res_type,
                score=score,
            )

    # ── Step 2: emit edges ───────────────────────────────────────────────────
    # Deduplicate within the same (from_name, to_name) pair — a concept may
    # call the same thing many times; we only want one edge per pair.
    seen: set[tuple[str, str, str]] = set()
    edges: list[CompositionEdge] = []

    for c in concepts:
        if c.name not in resolved:
            continue
        src = resolved[c.name]

        # Call-site edges
        for dep_name in set(c.calls):
            dep_pattern, _ = resolver.resolve_name(dep_name)
            if dep_pattern is None or dep_pattern == src.pattern:
                continue
            key = (src.name, dep_name, "calls")
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                CompositionEdge(
                    from_name=src.name,
                    from_pattern=src.pattern,
                    to_name=dep_name,
                    to_pattern=dep_pattern,
                    relationship="calls",
                    from_file=src.file_path,
                )
            )

        # Inheritance edges
        for base_name in c.bases:
            base_pattern, _ = resolver.resolve_name(base_name)
            if base_pattern is None or base_pattern == src.pattern:
                continue
            key = (src.name, base_name, "inherits")
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                CompositionEdge(
                    from_name=src.name,
                    from_pattern=src.pattern,
                    to_name=base_name,
                    to_pattern=base_pattern,
                    relationship="inherits",
                    from_file=src.file_path,
                )
            )

    return list(resolved.values()), edges

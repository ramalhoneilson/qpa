"""Maps concept names and docstrings to quantum patterns.

Loads all enriched knowledge-base CSV files and provides two resolution
strategies:

  resolve_name(name)        → (pattern, score) using embedding similarity on
                               the concept name.  Exact lowercase match is
                               tried first (score = 1.0) before embedding.

  resolve_docstring(text)   → (pattern, score) using embedding similarity on
                               a docstring / free-form description against the
                               KB summaries.  Used as a fallback when a defined
                               class name is not in the KB.

Both return (None, score) when the best match falls below the threshold.
"""
import csv
import re
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

# Enriched KB files (relative to project root)
_KB_FILES = [
    "data/knowledge_base/enriched_classiq_quantum_patterns.csv",
    "data/knowledge_base/enriched_pennylane_quantum_patterns.csv",
    "data/knowledge_base/enriched_qiskit_quantum_patterns.csv",
    "data/knowledge_base/enriched_qiskit_algorithms_quantum_patterns.csv",
]

# Thresholds — deliberately slightly higher than run_analysis.py to keep
# composition edges precise rather than noisy.
NAME_THRESHOLD = 0.92
DOCSTRING_THRESHOLD = 0.70


def _normalize(name: str) -> str:
    """Split CamelCase / snake_case into lowercase tokens for embedding."""
    name = name.replace("_", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    return name.lower().strip()


def _short_name(full_name: str) -> str:
    """Extract the last dotted / slash-separated segment."""
    return full_name.split("/")[-1].split(".")[-1]


class PatternResolver:
    """Loads the KB once and resolves concept names / docstrings to patterns."""

    def __init__(self, project_root: Path, model_name: str = "all-mpnet-base-v2"):
        self._exact: dict[str, str] = {}   # lowercase short name → pattern
        self._kb_names: list[str] = []
        self._kb_patterns: list[str] = []
        self._kb_summaries: list[str] = []

        for rel in _KB_FILES:
            path = project_root / rel
            if not path.exists():
                print(f"  [resolver] KB file not found, skipping: {path}")
                continue
            with open(path, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # header
                for row in reader:
                    if len(row) < 3:
                        continue
                    short = _short_name(row[0].strip())
                    pattern = row[2].strip()
                    summary = row[1].strip() if len(row) > 1 else ""
                    if short and pattern:
                        self._exact[short.lower()] = pattern
                        self._kb_names.append(short)
                        self._kb_patterns.append(pattern)
                        self._kb_summaries.append(summary)

        print(f"  [resolver] {len(self._kb_names)} KB entries loaded.")
        self._model = SentenceTransformer(model_name)
        self._name_emb = self._model.encode(
            [_normalize(n) for n in self._kb_names], convert_to_tensor=True
        )
        self._summary_emb = self._model.encode(
            self._kb_summaries, convert_to_tensor=True
        )

    def resolve_name(self, name: str) -> tuple[str | None, float]:
        """Return (pattern, score) for the best name match, or (None, score)."""
        if not name:
            return None, 0.0
        # Exact lookup (fast path)
        if name.lower() in self._exact:
            return self._exact[name.lower()], 1.0
        # Embedding similarity
        emb = self._model.encode([_normalize(name)], convert_to_tensor=True)
        sims = 1 - cdist(emb.cpu(), self._name_emb.cpu(), "cosine")[0]
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= NAME_THRESHOLD:
            return self._kb_patterns[idx], score
        return None, score

    def resolve_docstring(self, text: str) -> tuple[str | None, float]:
        """Return (pattern, score) by matching *text* against KB summaries."""
        if not text.strip():
            return None, 0.0
        emb = self._model.encode([text], convert_to_tensor=True)
        sims = 1 - cdist(emb.cpu(), self._summary_emb.cpu(), "cosine")[0]
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= DOCSTRING_THRESHOLD:
            return self._kb_patterns[idx], score
        return None, score

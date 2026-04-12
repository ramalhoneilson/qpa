"""
Extract public methods and module-level functions from qiskit-algorithms for
manual quantum-pattern review.

Output: data/qiskit_algorithms_methods_for_review.csv

Each row represents one method/function that passed the classical-noise filter.
The ``suggested_patterns`` column is pre-filled using keyword matching against
quantum_patterns.json — it is a *hint* for the manual review, not a definitive
classification.

Filtering (a method is DROPPED if ANY rule matches):
  - Name starts with ``_``  → dunder or private implementation detail
  - Decorated with ``@property``, ``@setter``, ``@deleter``,
    or ``@abstractproperty`` → property accessor / mutator
  - Name is in the classical-bookkeeping set (copy, to_dict, …)
"""

import ast
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from src.conf import config
from src.core_concepts.extractor.visitors import QiskitAlgorithmsMethodVisitor
from src.core_concepts.pipelines.extract_qiskit_algorithms import (
    QISKIT_ALGORITHMS_ROOT,
    _gather_source_files,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PATTERNS_JSON = config.PROJECT_ROOT / "data" / "quantum_patterns.json"
OUTPUT_CSV = config.RESULTS_DIR / "qiskit_algorithms_methods_for_review.csv"

# ---------------------------------------------------------------------------
# Pattern keyword builder
# ---------------------------------------------------------------------------

# Words too common in English or in quantum computing in general to be
# discriminating on their own.
_GENERIC_WORDS: set[str] = {
    "quantum", "the", "a", "an", "and", "or", "of", "for", "in",
    "to", "from", "with", "via", "using", "based", "on", "by", "is",
    "are", "as", "at", "be", "this", "that", "its", "also", "been",
    "has", "have", "not", "no", "can", "may", "will", "such", "each",
    "after", "into", "over", "same", "some", "than", "thus", "when",
    "which", "while", "within", "without", "how",
}


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alpha-numeric tokens from text."""
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9]+", text)]


def _phrases_from_name(name: str) -> list[str]:
    """
    Build a list of search phrases from a pattern name.

    Strategy:
    1. The full name as one phrase (best signal, low false-positive rate).
    2. Acronyms from parentheses, e.g. "(VQE)" → "vqe".

    E.g. "Quantum Phase Estimation (QPE)" → ["quantum phase estimation", "qpe"]
    """
    phrases = []

    # 1. Full name phrase (parens stripped).
    cleaned = re.sub(r"\(.*?\)", "", name).strip().lower()
    if cleaned:
        phrases.append(cleaned)

    # 2. Acronyms from parentheses.
    for acronym in re.findall(r"\(([A-Z]{2,})\)", name):
        phrases.append(acronym.lower())

    return list(dict.fromkeys(phrases))  # preserve order, drop duplicates


# Extra search terms added on top of the auto-generated ones, targeting the
# vocabulary actually used in qiskit-algorithms method names and docstrings.
# These are intentionally narrow to avoid false positives.
_EXTRA_KEYWORDS: dict[str, list[str]] = {
    "Amplitude Amplification": [
        "amplify",
        "amplification_problem",
        "grover_operator",
    ],
    "Oracle": [
        "is_good_state",
        "good_state",
        "oracle",
    ],
    "Initialization": [
        "prepare_state",
        "state_preparation",
        "initial_state",
        "prepare initial",
    ],
    "Variational Quantum Algorithm (VQA)": [
        "ansatz",
        "variational",
    ],
    "Variational Quantum Eigensolver (VQE)": [
        "minimum_eigenvalue",
        "compute_minimum_eigenvalue",
    ],
    "Uncompute": [
        "fidelity_circuit",
        "create_fidelity",
        "uncompute",
    ],
    "Angle Encoding": [
        # VQAs encode classical parameters as rotation angles; initial_point is
        # the array of rotation angles passed into the circuit.
        "initial_point",
        "rotation_angle",
    ],
    "Quantum Approximate Optimization Algorithm (QAOA)": [
        # QAOA-specific vocabulary visible in qaoa.py source and its parent VQE.
        "qaoa",
        "mixer",
    ],
    "Alternating Operator Ansatz (AOA)": [
        "mixer",
        "alternating",
    ],
    "Quantum Phase Estimation (QPE)": [
        "eigenphase",
        "phase_estimation",
        "estimate_from_pe",
    ],
    "Grover": [
        "grover",
        "optimal_num_iterations",
    ],
}


def build_pattern_keyword_map(patterns_json_path: Path) -> dict[str, list[str]]:
    """
    Return ``{pattern_name: [search_phrases]}``.

    Each phrase is checked as a *substring* (not whole-word) in the combined
    text of method name + docstring + source code to keep false-negative rate low.
    Short / ambiguous phrases are excluded to limit false positives.
    """
    with open(patterns_json_path, encoding="utf-8") as fh:
        patterns = json.load(fh)

    result: dict[str, list[str]] = {}
    for p in patterns:
        name: str = p["name"]
        alias: str = p.get("alias", "")

        phrases: list[str] = _phrases_from_name(name)

        # Add alias phrase if it looks meaningful (not a placeholder).
        _alias_junk = {"—", "–", "", "not available"}
        if (
            alias
            and alias.lower() not in _alias_junk
            and "enter your input" not in alias.lower()
            and len(alias) < 100
        ):
            phrases += _phrases_from_name(alias)

        # Drop phrases that are only generic words or too short to be useful.
        meaningful = []
        for phrase in phrases:
            tokens = _tokenize(phrase)
            non_generic = [t for t in tokens if t not in _GENERIC_WORDS and len(t) >= 3]
            if non_generic:
                meaningful.append(phrase)

        # Merge in any hand-curated extras for this pattern.
        meaningful += _EXTRA_KEYWORDS.get(name, [])

        result[name] = list(dict.fromkeys(meaningful))  # deduplicate, preserve order

    return result


def match_patterns(
    text: str, keyword_map: dict[str, list[str]]
) -> list[str]:
    """
    Return list of pattern names whose search phrases appear in *text*.
    *text* is the combined method name + docstring + source code (lower-cased).
    """
    text_lower = text.lower()
    matched = []
    for pattern_name, phrases in keyword_map.items():
        if any(phrase in text_lower for phrase in phrases):
            matched.append(pattern_name)
    return matched


# ---------------------------------------------------------------------------
# File-level extractor
# ---------------------------------------------------------------------------

def _extract_methods_from_file(
    py_file: Path,
) -> list[dict[str, Any]]:
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        visitor = QiskitAlgorithmsMethodVisitor(
            source_text=source,
            file_path=py_file,
            sdk_root=QISKIT_ALGORITHMS_ROOT,
        )
        visitor.visit(tree)
        return visitor.found_methods
    except Exception as exc:
        logging.warning(f"Could not parse {py_file.name}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.info("--- Starting qiskit-algorithms Method Extraction for Review ---")

    if not QISKIT_ALGORITHMS_ROOT.is_dir():
        logging.error(
            f"qiskit-algorithms root not found: {QISKIT_ALGORITHMS_ROOT.resolve()}"
        )
        return

    if not PATTERNS_JSON.exists():
        logging.error(f"quantum_patterns.json not found: {PATTERNS_JSON}")
        return

    # 1. Build the pattern keyword map.
    keyword_map = build_pattern_keyword_map(PATTERNS_JSON)
    logging.info(f"Loaded {len(keyword_map)} patterns for keyword matching.")

    # 2. Gather source files (same whitelist as the class extractor).
    source_files = _gather_source_files()
    logging.info(f"Scanning {len(source_files)} source files.")

    # 3. Extract methods.
    all_methods: list[dict[str, Any]] = []
    for py_file in source_files:
        all_methods.extend(_extract_methods_from_file(py_file))

    logging.info(f"Extracted {len(all_methods)} methods/functions after filtering.")

    # 4. Match each method against the pattern keyword map.
    #    The search corpus is: method_name + docstring_summary + source_code.
    rows: list[dict[str, str]] = []
    for m in all_methods:
        corpus = " ".join([
            m["method_name"],
            m["docstring_summary"],
            m["source_code"],
        ])
        matched = match_patterns(corpus, keyword_map)
        rows.append(
            {
                "file_path": m["file_path"],
                "class_name": m["class_name"],
                "method_name": m["method_name"],
                "docstring_summary": m["docstring_summary"],
                "suggested_patterns": "; ".join(matched),
            }
        )

    # 5. Sort by file then class then method for readability.
    rows.sort(key=lambda r: (r["file_path"], r["class_name"], r["method_name"]))

    # 6. Write CSV.
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file_path",
                "class_name",
                "method_name",
                "docstring_summary",
                "suggested_patterns",
            ],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)

    matched_count = sum(1 for r in rows if r["suggested_patterns"])
    logging.info(
        f"Wrote {len(rows)} rows to '{OUTPUT_CSV}' "
        f"({matched_count} with at least one suggested pattern, "
        f"{len(rows) - matched_count} flagged for manual review with no auto-match)."
    )
    logging.info("--- Method Extraction Complete ---")


if __name__ == "__main__":
    main()

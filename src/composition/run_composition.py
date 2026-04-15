"""CLI entry point for the static composition analysis.

Scans a target directory for Python files, extracts every class and top-level
function definition, resolves each to a quantum pattern, then builds a
directed composition graph showing which patterns depend on which.

Usage
-----
    python -m src.composition.run_composition --target-dir <path>

    # optional: custom output directory
    python -m src.composition.run_composition \\
        --target-dir target_github_projects/Qrisp \\
        --output-dir data/composition_output/qrisp

Outputs (written to --output-dir, default: data/composition_output/)
---------------------------------------------------------------------
  composition_concepts.csv   Every resolved concept → pattern
  composition_edges.csv      Raw concept-level directed edges
  composition_patterns.csv   Aggregated pattern-level edges with edge counts
  composition_diagram.md     Mermaid flowchart of the pattern-level graph
"""
import argparse
import collections
import csv
from pathlib import Path

from src.conf import config
from src.composition.call_graph import extract_defined_concepts
from src.composition.pattern_resolver import PatternResolver
from src.composition.composition_graph import build_composition_graph


# ── helpers ──────────────────────────────────────────────────────────────────

def _iter_python_files(target_dir: Path):
    """Yield .py files, skipping directories listed in config.SKIP_DIRS_COMMON."""
    skip = set(config.SKIP_DIRS_COMMON)
    skip_files = set(config.SKIP_FILES_COMMON)
    for path in sorted(target_dir.rglob("*.py")):
        if any(part in skip for part in path.parts):
            continue
        if path.name in skip_files:
            continue
        yield path


def _write_mermaid(
    pattern_edges: dict[tuple[str, str, str], int],
    output_path: Path,
    min_count: int = 1,
):
    """Write a Mermaid flowchart of the pattern-level composition graph."""
    # Shorten long pattern names for readability inside diagram nodes
    def _short(p: str) -> str:
        replacements = {
            "Variational Quantum Eigensolver (VQE)": "VQE",
            "Variational Quantum Algorithm (VQA)": "VQA",
            "Quantum Approximate Optimization Algorithm (QAOA)": "QAOA",
            "Quantum Amplitude Estimation": "QAE",
            "Quantum Phase Estimation (QPE)": "QPE",
            "Linear Combination of Unitaries": "LCU",
            "Amplitude Amplification": "Amplitude Amplif.",
            "Hamiltonian Simulation": "Hamiltonian Sim.",
            "Circuit Construction Utility": "CCU",
            "Domain Specific Application": "DSA",
        }
        return replacements.get(p, p)

    lines = ["# Composition Graph — Pattern Level", "", "```mermaid", "flowchart TD"]

    # Collect nodes
    nodes: set[str] = set()
    for (fp, tp, _), cnt in pattern_edges.items():
        if cnt >= min_count:
            nodes.add(fp)
            nodes.add(tp)

    # Node id map (mermaid ids cannot have spaces/parens)
    node_id = {p: f"N{i}" for i, p in enumerate(sorted(nodes))}

    for p, nid in node_id.items():
        lines.append(f'    {nid}["{_short(p)}"]')

    lines.append("")

    # Edges
    for (fp, tp, rel), cnt in sorted(pattern_edges.items(), key=lambda x: -x[1]):
        if cnt < min_count:
            continue
        arrow = "-->" if rel == "calls" else "-.->"
        label = f"|{rel} x{cnt}|"
        lines.append(f"    {node_id[fp]} {arrow}{label} {node_id[tp]}")

    lines.append("```")
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ── main pipeline ─────────────────────────────────────────────────────────────

def main(target_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect all Python files
    print(f"\nScanning: {target_dir}")
    py_files = list(_iter_python_files(target_dir))
    print(f"  {len(py_files)} Python files found.")

    # 2. Extract defined concepts (class + top-level function definitions)
    all_concepts = []
    for f in py_files:
        all_concepts.extend(extract_defined_concepts(f))
    print(f"  {len(all_concepts)} class/function definitions extracted.")

    # 3. Load KB + embedding model
    print("\nLoading pattern resolver ...")
    resolver = PatternResolver(config.PROJECT_ROOT)

    # 4. Build composition graph
    print("\nBuilding composition graph ...")
    resolved, edges = build_composition_graph(all_concepts, resolver)
    print(f"  {len(resolved)} concepts resolved to quantum patterns.")
    print(f"  {len(edges)} directed composition edges found.")

    # 5. Write composition_concepts.csv
    concepts_path = output_dir / "composition_concepts.csv"
    with open(concepts_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept_name", "pattern", "resolution_type", "score", "file_path"])
        for r in sorted(resolved, key=lambda x: (x.pattern, x.name)):
            w.writerow([r.name, r.pattern, r.resolution_type, f"{r.score:.4f}", r.file_path])
    print(f"  → {concepts_path}")

    # 6. Write composition_edges.csv (concept level)
    edges_path = output_dir / "composition_edges.csv"
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["from_concept", "from_pattern", "to_concept", "to_pattern",
             "relationship", "file_path"]
        )
        for e in sorted(edges, key=lambda x: (x.from_pattern, x.to_pattern)):
            w.writerow(
                [e.from_name, e.from_pattern, e.to_name, e.to_pattern,
                 e.relationship, e.from_file]
            )
    print(f"  → {edges_path}")

    # 7. Aggregate to pattern level
    pattern_edges: collections.Counter = collections.Counter()
    for e in edges:
        pattern_edges[(e.from_pattern, e.to_pattern, e.relationship)] += 1

    patterns_path = output_dir / "composition_patterns.csv"
    with open(patterns_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from_pattern", "to_pattern", "relationship", "edge_count"])
        for (fp, tp, rel), cnt in sorted(pattern_edges.items(), key=lambda x: -x[1]):
            w.writerow([fp, tp, rel, cnt])
    print(f"  → {patterns_path}")

    # 8. Mermaid diagram
    diagram_path = output_dir / "composition_diagram.md"
    _write_mermaid(pattern_edges, diagram_path)
    print(f"  → {diagram_path}")

    # 9. Summary to stdout
    print("\n--- TOP COMPOSITION RELATIONSHIPS ---")
    for (fp, tp, rel), cnt in sorted(pattern_edges.items(), key=lambda x: -x[1])[:25]:
        arrow = "→" if rel == "calls" else "⇢"
        print(f"  {fp}  {arrow}[{rel}]  {tp}   ({cnt}x)")

    print(
        f"\nDone.  {len(resolved)} concepts, {len(edges)} edges, "
        f"{len(pattern_edges)} unique pattern-level relationships."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static call-graph composition analysis: identifies which quantum "
            "patterns are built from other quantum patterns."
        )
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Root directory of the project to analyse (scanned recursively).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for output files.  "
            "Default: data/composition_output/<target-dir-name>/"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    target = Path(args.target_dir).resolve()
    if args.output_dir:
        out = Path(args.output_dir).resolve()
    else:
        out = config.PROJECT_ROOT / "data" / "composition_output" / target.name
    main(target, out)

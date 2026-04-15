"""
Compare embedding model performance using ChromaDB as the vector store.

For each model listed in ``indexer.MODELS``:
1. Build ChromaDB collections (skip if already indexed).
2. Run semantic pattern matching on a target project directory.
3. Compute precision / recall / F1 against the project's ground truth.
4. Print a side-by-side comparison table and save a Markdown report.

The matching logic mirrors ``src/analysis/run_analysis.py`` exactly — same
thresholds, same normalisation, same deduplication — so differences in the
results are due to the embedding model alone.

Usage
-----
    # Compare on qiskit-finance (smallest GT, fastest to run):
    python -m src.evaluation.embedding_comparison.run_comparison

    # Compare on a different evaluation project:
    python -m src.evaluation.embedding_comparison.run_comparison --project qualtran
    python -m src.evaluation.embedding_comparison.run_comparison --project qrisp

    # Force-rebuild ChromaDB collections first (after KB changes):
    python -m src.evaluation.embedding_comparison.run_comparison --rebuild
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

from src.analysis.run_analysis import (
    CONCEPT_FILES,
    PATTERN_FILES,
    extract_comments_from_script,
    load_patterns_map,
    load_quantum_concepts,
    normalize_identifier,
)
from src.conf import config
from src.evaluation.embedding_comparison.indexer import (
    MODELS,
    build_collections,
    model_slug,
)
from src.evaluation.metrics import AggregateMetrics, compute_metrics, join_predictions

# ── Evaluation project registry ────────────────────────────────────────────────

PROJECTS: dict[str, dict] = {
    "qiskit-finance": {
        "target_dir": config.PROJECT_ROOT / "target_github_projects/qiskit-finance",
        "gt_path": config.RESULTS_DIR / "qiskit_finance_ground_truth.csv",
    },
    "qrisp": {
        "target_dir": config.PROJECT_ROOT / "target_github_projects/Qrisp",
        "gt_path": config.RESULTS_DIR / "qrisp_ground_truth.csv",
    },
    "qualtran": {
        "target_dir": config.PROJECT_ROOT / "target_github_projects/Qualtran",
        "gt_path": config.RESULTS_DIR / "qualtran_ground_truth.csv",
    },
}

# ── Matching thresholds — must mirror run_analysis.py ─────────────────────────
# ChromaDB uses cosine *distance* = 1 − cosine_similarity, so the direction
# of the comparison is flipped relative to the similarity threshold.

_SIM_THRESHOLDS = {"name": 0.90, "summary": 0.65}
_DIST_THRESHOLDS = {k: round(1.0 - v, 10) for k, v in _SIM_THRESHOLDS.items()}

# How many top candidates to fetch per query before applying the distance
# threshold.  Larger values are safer (won't miss matches) but slower.
# At threshold 0.90 there are at most 1–3 name matches per element in
# practice; at 0.65 summaries rarely exceed 20 matches per file.
_N_RESULTS_NAMES = 20
_N_RESULTS_SUMMARIES = 50

# ── Output paths ───────────────────────────────────────────────────────────────
_OUTPUT_DIR = config.RESULTS_DIR / "embedding_comparison"
_REPORT_PATH = config.DOCS_DIR / "embedding_comparison.md"


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model_name: str
    agg: AggregateMetrics
    index_time_s: float   # time to build / load collections
    query_time_s: float   # time to scan all files
    total_matches: int    # unique (file, concept) pairs found


# ── Analysis logic ─────────────────────────────────────────────────────────────

def _run_analysis_with_chroma(
    target_dir: Path,
    model_name: str,
    concepts: list[dict],
    force_rebuild: bool = False,
) -> tuple[list[dict], float, float]:
    """Run semantic pattern matching for *model_name* using ChromaDB.

    Parameters
    ----------
    target_dir:
        Root directory of the target project to scan.
    model_name:
        HuggingFace model identifier.
    concepts:
        Pre-loaded KB concept list.
    force_rebuild:
        Rebuild ChromaDB collections from scratch.

    Returns
    -------
    (rows, index_time_s, query_time_s)
        *rows* has the same schema as ``run_analysis.py`` output rows.
    """
    t0 = time.perf_counter()
    names_coll, summaries_coll = build_collections(
        model_name, concepts, force_rebuild=force_rebuild
    )
    index_time = time.perf_counter() - t0

    name_dist_thresh = _DIST_THRESHOLDS["name"]
    summary_dist_thresh = _DIST_THRESHOLDS["summary"]

    script_files = list(target_dir.rglob("*.py"))
    total_files = len(script_files)

    # key: (relative_path, concept_name) → best-scoring match dict
    best_matches: dict[tuple[str, str], dict] = {}

    t1 = time.perf_counter()
    for i, file_path in enumerate(script_files):
        if (i + 1) % 200 == 0 or (i + 1) == total_files:
            print(f"    [{model_name}] {i + 1}/{total_files} files...")

        try:
            script_content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        relative_path = str(file_path.relative_to(target_dir))

        # ── Name-based matching ──────────────────────────────────────────────
        located_elements: list[tuple[str, int]] = []
        try:
            tree = ast.parse(script_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name):
                        located_elements.append((func.id, node.lineno))
                    elif isinstance(func, ast.Attribute):
                        located_elements.append((func.attr, node.lineno))
        except SyntaxError:
            pass

        if located_elements:
            query_texts = [normalize_identifier(e[0]) for e in located_elements]
            results = names_coll.query(
                query_texts=query_texts,
                n_results=min(_N_RESULTS_NAMES, names_coll.count()),
                include=["distances", "metadatas"],
            )
            for elem_idx, (element, lineno) in enumerate(located_elements):
                dists = results["distances"][elem_idx]
                metas = results["metadatas"][elem_idx]
                for dist, meta in zip(dists, metas):
                    # Results are sorted ascending by distance; stop early.
                    if dist > name_dist_thresh:
                        break
                    score = 1.0 - dist
                    key = (relative_path, meta["concept_name"])
                    if key not in best_matches or score > best_matches[key]["score"]:
                        best_matches[key] = {
                            "file_path": relative_path,
                            "concept_name": meta["concept_name"],
                            "pattern": meta["pattern"],
                            "match_type": "name",
                            "matched_text": element,
                            "score": score,
                            "first_line": lineno,
                        }

        # ── Summary-based matching ───────────────────────────────────────────
        comment_block = extract_comments_from_script(file_path)
        if comment_block:
            results = summaries_coll.query(
                query_texts=[comment_block],
                n_results=min(_N_RESULTS_SUMMARIES, summaries_coll.count()),
                include=["distances", "metadatas"],
            )
            dists = results["distances"][0]
            metas = results["metadatas"][0]
            truncated = (
                (comment_block[:150] + "...") if len(comment_block) > 150 else comment_block
            )
            for dist, meta in zip(dists, metas):
                if dist > summary_dist_thresh:
                    break
                score = 1.0 - dist
                key = (relative_path, meta["concept_name"])
                if key not in best_matches or score > best_matches[key]["score"]:
                    best_matches[key] = {
                        "file_path": relative_path,
                        "concept_name": meta["concept_name"],
                        "pattern": meta["pattern"],
                        "match_type": "summary",
                        "matched_text": truncated.replace(";", ","),
                        "score": score,
                        "first_line": 0,
                    }

    query_time = time.perf_counter() - t1
    return list(best_matches.values()), index_time, query_time


def _save_analysis_output(rows: list[dict], output_path: Path) -> None:
    """Write analysis rows to a semicolon-delimited CSV (same format as run_analysis.py)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: (r["file_path"], r["first_line"]))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            ["file_path", "concept_name", "pattern", "match_type",
             "matched_text", "similarity_score", "first_line"]
        )
        for row in sorted_rows:
            writer.writerow([
                row["file_path"],
                row["concept_name"],
                row["pattern"],
                row["match_type"],
                row["matched_text"],
                f"{row['score']:.4f}",
                row["first_line"],
            ])


# ── Reporting ──────────────────────────────────────────────────────────────────

def _f(v: float) -> str:
    return f"{v:.3f}" if not math.isnan(v) else "  — "


def _print_comparison(results: list[ModelResult], project: str) -> None:
    print(f"\n{'=' * 105}")
    print(f"  Embedding Model Comparison — Project: {project}")
    print(f"{'=' * 105}")
    header = (
        f"  {'Model':<45}"
        f" {'µP':>6} {'µR':>6} {'µF1':>6}"
        f" {'MP':>6} {'MR':>6} {'MF1':>6}"
        f" {'Matches':>8} {'Idx(s)':>7} {'Qry(s)':>7}"
    )
    print(header)
    print(f"  {'-' * 103}")
    for r in results:
        a = r.agg
        print(
            f"  {r.model_name:<45}"
            f" {_f(a.micro_precision):>6} {_f(a.micro_recall):>6} {_f(a.micro_f1):>6}"
            f" {_f(a.macro_precision):>6} {_f(a.macro_recall):>6} {_f(a.macro_f1):>6}"
            f" {r.total_matches:>8}"
            f" {r.index_time_s:>7.1f}"
            f" {r.query_time_s:>7.1f}"
        )
    print()


def _save_report(results: list[ModelResult], project: str) -> None:
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Embedding Model Comparison — {project}\n",
        "| Model | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1"
        " | Matches | Index (s) | Query (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        a = r.agg
        lines.append(
            f"| {r.model_name}"
            f" | {_f(a.micro_precision)} | {_f(a.micro_recall)} | {_f(a.micro_f1)}"
            f" | {_f(a.macro_precision)} | {_f(a.macro_recall)} | {_f(a.macro_f1)}"
            f" | {r.total_matches} | {r.index_time_s:.1f} | {r.query_time_s:.1f} |"
        )
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved to {_REPORT_PATH}")


# ── Main comparison orchestrator ───────────────────────────────────────────────

def compare(project: str = "qiskit-finance", rebuild: bool = False) -> list[ModelResult]:
    """Run the full comparison for all models on *project*.

    Parameters
    ----------
    project:
        Key in ``PROJECTS`` — one of ``"qiskit-finance"``, ``"qrisp"``,
        ``"qualtran"``.
    rebuild:
        Force-rebuild ChromaDB collections before running.

    Returns
    -------
    List of ``ModelResult`` objects, one per model.
    """
    if project not in PROJECTS:
        raise ValueError(f"Unknown project '{project}'. Choose from: {list(PROJECTS)}")

    cfg = PROJECTS[project]
    target_dir: Path = cfg["target_dir"]
    gt_path: Path = cfg["gt_path"]

    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_path}")

    print("Loading KB concepts...")
    pattern_map = load_patterns_map(PATTERN_FILES)
    concepts = load_quantum_concepts(CONCEPT_FILES, pattern_map)
    print(f"Loaded {len(concepts)} concepts.\n")

    results: list[ModelResult] = []

    for model_name in MODELS:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"{'─' * 60}")

        output_path = _OUTPUT_DIR / project / f"{model_slug(model_name)}_output.csv"

        rows, idx_t, qry_t = _run_analysis_with_chroma(
            target_dir, model_name, concepts, force_rebuild=rebuild
        )
        _save_analysis_output(rows, output_path)

        # join_predictions reads from disk, so we pass string paths.
        eval_df = join_predictions(str(gt_path), str(output_path))
        _, agg = compute_metrics(eval_df)

        results.append(ModelResult(
            model_name=model_name,
            agg=agg,
            index_time_s=idx_t,
            query_time_s=qry_t,
            total_matches=len(rows),
        ))
        print(
            f"  Micro F1: {agg.micro_f1:.3f}"
            f"  (P={agg.micro_precision:.3f}, R={agg.micro_recall:.3f})"
            f"  matches={len(rows)}"
        )

    _print_comparison(results, project)
    _save_report(results, project)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare embedding models for quantum pattern matching."
    )
    parser.add_argument(
        "--project",
        choices=list(PROJECTS),
        default="qiskit-finance",
        help="Evaluation project to run against (default: qiskit-finance).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe and rebuild ChromaDB collections before running.",
    )
    args = parser.parse_args()
    compare(project=args.project, rebuild=args.rebuild)


if __name__ == "__main__":
    main()

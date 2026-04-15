"""
End-to-end evaluation pipeline.

1. Runs run_analysis.py on a target directory (or reuses existing output).
2. Joins predictions against a ground truth CSV.
3. Computes precision / recall / F1.
4. Writes the updated ground truth CSV and a markdown report.
5. Prints a summary table to stdout.

Usage (module)
--------------
    python -m src.evaluation.evaluate \\
        --project    qiskit-finance \\
        --target-dir target_github_projects/qiskit-finance \\
        --gt         data/qiskit_finance_ground_truth.csv \\
        --output     data/qiskit_finance_eval_output.csv \\
        --report     docs/qiskit_finance_precision_recall.md

    # Skip re-running the analysis (reuse an existing output CSV):
    python -m src.evaluation.evaluate ... --skip-analysis
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import AggregateMetrics, PatternMetrics, compute_metrics, join_predictions
from src.evaluation.report import generate_report


def _run_analysis(target_dir: str, output_path: str) -> None:
    cmd = [
        sys.executable, "-m", "src.analysis.run_analysis",
        "--target-dir", target_dir,
        "--output", output_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parents[2])
    if result.returncode != 0:
        raise RuntimeError(f"run_analysis.py failed with exit code {result.returncode}")


def _print_summary(
    project: str,
    agg: AggregateMetrics,
    per_pattern: list[PatternMetrics],
) -> None:
    def _f(v: float) -> str:
        return f"{v:.3f}" if not math.isnan(v) else "  — "

    print(f"\n{'=' * 80}")
    print(f"  Evaluation: {project}")
    print(f"{'=' * 80}")
    print(f"  GT files:          {agg.total_gt}")
    print(f"  Predicted:         {agg.total_predicted} ({agg.total_predicted / agg.total_gt * 100:.0f} %)")
    print(f"  Missed:            {agg.total_missed}")
    print(f"  Correct (TP):      {agg.total_tp}")
    print(f"  Micro  P/R/F1:     {agg.micro_precision:.3f} / {agg.micro_recall:.3f} / {agg.micro_f1:.3f}")
    print(f"  Macro  P/R/F1:     {agg.macro_precision:.3f} / {agg.macro_recall:.3f} / {agg.macro_f1:.3f}")
    print()
    print(f"  {'Pattern':<45} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-' * 97}")
    for m in sorted(per_pattern, key=lambda x: x.gt_count, reverse=True):
        print(
            f"  {m.pattern:<45} {m.gt_count:>4} {m.pred_count:>5} {m.tp:>4} {m.fp:>4} {m.fn:>4}"
            f" {_f(m.precision):>6} {_f(m.recall):>6} {_f(m.f1):>6}"
        )
    print()


def evaluate(
    project: str,
    target_dir: str,
    gt_path: str,
    output_path: str,
    report_path: str,
    skip_analysis: bool = False,
) -> tuple[pd.DataFrame, list[PatternMetrics], AggregateMetrics]:
    if not skip_analysis:
        _run_analysis(target_dir, output_path)

    eval_df = join_predictions(gt_path, output_path)
    per_pattern, agg = compute_metrics(eval_df)

    # Persist updated ground truth
    eval_df.to_csv(gt_path, index=False)

    generate_report(project, target_dir, eval_df, per_pattern, agg, report_path)
    _print_summary(project, agg, per_pattern)

    return eval_df, per_pattern, agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pattern-finder precision/recall.")
    parser.add_argument("--project",       required=True, help="Human-readable project name")
    parser.add_argument("--target-dir",    required=True, help="Path to target project directory")
    parser.add_argument("--gt",            required=True, help="Path to ground truth CSV")
    parser.add_argument("--output",        required=True, help="Path for analysis output CSV")
    parser.add_argument("--report",        required=True, help="Path for markdown report")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip running run_analysis.py (reuse existing output CSV)")
    args = parser.parse_args()

    evaluate(
        project=args.project,
        target_dir=args.target_dir,
        gt_path=args.gt,
        output_path=args.output,
        report_path=args.report,
        skip_analysis=args.skip_analysis,
    )


if __name__ == "__main__":
    main()

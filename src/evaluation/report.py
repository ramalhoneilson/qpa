"""
Markdown report generator for precision/recall evaluation results.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import AggregateMetrics, PatternMetrics


def _fmt(v: float, pct: bool = False) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:.3f}" if not pct else f"{v * 100:.1f} %"


def generate_report(
    project_name: str,
    target_dir: str,
    eval_df: pd.DataFrame,
    per_pattern: list[PatternMetrics],
    agg: AggregateMetrics,
    output_path: str,
) -> None:
    """Write a markdown precision/recall report to output_path."""

    correct_rows = eval_df[eval_df["match"] == 1][["file_path", "ground_truth_pattern"]]
    wrong_rows   = eval_df[(eval_df["predicted_pattern"].notna()) & (eval_df["match"] == 0)][
        ["file_path", "ground_truth_pattern", "predicted_pattern"]
    ]

    lines: list[str] = []
    lines += [
        f"# Precision and Recall Evaluation — {project_name}",
        "",
        "## Setup",
        "",
        f"- **Target project**: `{target_dir}`",
        f"- **Ground truth**: `data/{project_name.lower().replace('-', '_')}_ground_truth.csv`"
        f" ({agg.total_gt} files, {eval_df['ground_truth_pattern'].nunique()} patterns)",
        "- **Predictor**: `run_analysis.py --target-dir`",
        "- **Resolution**: highest-scoring prediction per file; ties broken alphabetically by pattern.",
        "",
        "---",
        "",
        "## Overall Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| GT files | {agg.total_gt} |",
        f"| Files with at least one prediction | {agg.total_predicted} ({agg.total_predicted / agg.total_gt * 100:.0f} %) |",
        f"| Files with no prediction (missed) | {agg.total_missed} ({agg.total_missed / agg.total_gt * 100:.0f} %) |",
        f"| Correct predictions (TP) | {agg.total_tp} |",
        f"| Wrong predictions (FP) | {agg.total_predicted - agg.total_tp} |",
        f"| **Micro Precision** | **{agg.micro_precision:.3f}** |",
        f"| **Micro Recall** | **{agg.micro_recall:.3f}** |",
        f"| **Micro F1** | **{agg.micro_f1:.3f}** |",
        f"| Macro Precision | {agg.macro_precision:.3f} |",
        f"| Macro Recall | {agg.macro_recall:.3f} |",
        f"| Macro F1 | {agg.macro_f1:.3f} |",
        "",
        "---",
        "",
        "## Per-Pattern Results",
        "",
        "| Pattern | GT | Predicted | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in sorted(per_pattern, key=lambda x: x.gt_count, reverse=True):
        lines.append(
            f"| {m.pattern} | {m.gt_count} | {m.pred_count} | {m.tp} | {m.fp} | {m.fn}"
            f" | {_fmt(m.precision)} | {_fmt(m.recall)} | {_fmt(m.f1)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Correct Predictions",
        "",
    ]
    if correct_rows.empty:
        lines.append("_None._")
    else:
        lines.append("| File | Pattern |")
        lines.append("|---|---|")
        for _, r in correct_rows.iterrows():
            lines.append(f"| `{r['file_path']}` | {r['ground_truth_pattern']} |")

    lines += [
        "",
        "---",
        "",
        "## Wrong Predictions",
        "",
    ]
    if wrong_rows.empty:
        lines.append("_None._")
    else:
        lines.append("| File | GT Pattern | Predicted Pattern |")
        lines.append("|---|---|---|")
        for _, r in wrong_rows.iterrows():
            lines.append(f"| `{r['file_path']}` | {r['ground_truth_pattern']} | {r['predicted_pattern']} |")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output_path}")

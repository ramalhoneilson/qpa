"""
Per-framework precision / recall / observed-recall evaluation for:
  - Amazon Braket Algorithm Library (19 notebooks)
  - Qiskit Machine Learning (13 tutorial notebooks)

Observed recall = recall computed only over patterns that have at least one
ground-truth positive in the framework being evaluated.  Patterns absent from
the framework's GT do not contribute a denominator, so specialisation cannot
artificially reduce the score.

Two evaluation modes per framework:
  single_label  – keep only the best-scoring prediction per file (alphabetical
                  tiebreak); classic single-label evaluation.
  multi_label   – credit a TP whenever the GT pattern appears in ANY of the
                  file's predictions above threshold; appropriate for QML
                  notebooks that legitimately contain multiple patterns.

Usage
-----
    python -m src.evaluation.pipelines.evaluate_braket_and_qiskit_ml
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[3]
DATA = PROJECT_ROOT / "data"

CONFIGS = [
    {
        "name": "Amazon Braket Algorithm Library",
        "gt":     DATA / "braket_algorithm_library_ground_truth.csv",
        "pred":   DATA / "braket_algorithm_library_eval_output.csv",
        "report": PROJECT_ROOT / "docs" / "braket_precision_recall.md",
        "multi_label": False,
    },
    {
        "name": "Qiskit Machine Learning",
        "gt":     DATA / "qiskit_machine_learning_ground_truth.csv",
        "pred":   DATA / "qiskit_machine_learning_eval_output.csv",
        "report": PROJECT_ROOT / "docs" / "qiskit_ml_precision_recall.md",
        "multi_label": True,
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PatternResult:
    pattern: str
    gt_count: int = 0
    pred_count: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")


@dataclass
class FrameworkResult:
    name: str
    total_gt: int
    total_predicted: int
    total_tp: int
    total_missed: int
    multi_label: bool
    # Standard micro metrics (over all files)
    micro_precision: float
    micro_recall: float
    micro_f1: float
    # Observed macro metrics (averaged only over patterns present in GT)
    observed_macro_recall: float
    observed_macro_precision: float
    observed_macro_f1: float
    per_pattern: list[PatternResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _nan(v: float) -> str:
    return f"{v:.3f}" if not math.isnan(v) else "  —  "


def join_and_evaluate(
    gt_path: Path,
    pred_path: Path,
    multi_label: bool = False,
) -> tuple[pd.DataFrame, list[PatternResult], FrameworkResult]:
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path, sep=";")

    # Drop stale columns from GT
    drop_cols = [c for c in ["predicted_pattern", "pred_score", "match",
                              "all_predicted_patterns", "gt_detected"] if c in gt.columns]
    gt = gt.drop(columns=drop_cols)

    # Keep only predictions for files that appear in GT
    pred_in_gt = pred[pred["file_path"].isin(gt["file_path"])].copy()

    if multi_label:
        # ── Multi-label mode ──────────────────────────────────────────────────
        # For each file, collect the set of ALL distinct patterns predicted
        # above threshold (everything in pred_in_gt already passed threshold).
        all_preds_per_file = (
            pred_in_gt.groupby("file_path")["pattern"]
            .apply(lambda s: set(s.tolist()))
            .reset_index()
            .rename(columns={"pattern": "all_predicted_patterns"})
        )

        # Best single prediction for display purposes
        pred_best = (
            pred_in_gt
            .sort_values(["file_path", "similarity_score", "pattern"], ascending=[True, False, True])
            .drop_duplicates(subset=["file_path"], keep="first")
            [["file_path", "pattern", "similarity_score"]]
            .rename(columns={"pattern": "top_predicted_pattern", "similarity_score": "pred_score"})
        )

        df = gt.merge(all_preds_per_file, on="file_path", how="left")
        df = df.merge(pred_best, on="file_path", how="left")

        # GT detected = GT pattern appears anywhere in the prediction set
        df["gt_detected"] = df.apply(
            lambda r: (
                isinstance(r.get("all_predicted_patterns"), set)
                and r["ground_truth_pattern"] in r["all_predicted_patterns"]
            ),
            axis=1,
        )

        # Per-pattern metrics (multi-label TP/FP/FN)
        all_patterns = sorted(df["ground_truth_pattern"].unique())
        per_pattern: list[PatternResult] = []

        for p in all_patterns:
            gt_p = df["ground_truth_pattern"] == p

            # TP: GT=p AND p was predicted for that file
            tp_mask = gt_p & df["gt_detected"]
            tp = int(tp_mask.sum())

            # FN: GT=p AND p was NOT predicted
            fn = int((gt_p & ~df["gt_detected"]).sum())

            # FP: GT≠p AND p appears in the prediction set
            fp_mask = ~gt_p & df["all_predicted_patterns"].apply(
                lambda s: p in s if isinstance(s, set) else False
            )
            fp = int(fp_mask.sum())

            # pred_count: files where p appears in any prediction
            pred_count = int(df["all_predicted_patterns"].apply(
                lambda s: p in s if isinstance(s, set) else False
            ).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1   = (2 * prec * rec / (prec + rec)
                    if (not math.isnan(prec) and not math.isnan(rec) and (prec + rec) > 0)
                    else float("nan"))

            per_pattern.append(PatternResult(
                pattern=p,
                gt_count=int(gt_p.sum()),
                pred_count=pred_count,
                tp=tp, fp=fp, fn=fn,
                precision=prec, recall=rec, f1=f1,
            ))

        # Aggregate
        total_tp   = sum(m.tp for m in per_pattern)
        total_pred = int(df["all_predicted_patterns"].notna().sum())
        total_gt   = len(df)
        total_miss = int(df["all_predicted_patterns"].isna().sum())

        # Micro P/R: TP over all GT files; for precision divide by files that
        # returned at least one prediction.
        micro_p  = total_tp / total_pred if total_pred > 0 else 0.0
        micro_r  = total_tp / total_gt   if total_gt  > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) > 0 else 0.0)

        # Re-alias for the file-level table (match column)
        df["match"] = df["gt_detected"].apply(lambda v: 1 if v else 0)

    else:
        # ── Single-label mode (original behaviour) ────────────────────────────
        pred_best = (
            pred_in_gt
            .sort_values(["file_path", "similarity_score", "pattern"], ascending=[True, False, True])
            .drop_duplicates(subset=["file_path"], keep="first")
            [["file_path", "pattern", "similarity_score"]]
            .rename(columns={"pattern": "predicted_pattern", "similarity_score": "pred_score"})
        )

        df = gt.merge(pred_best, on="file_path", how="left")
        df["match"] = df.apply(
            lambda r: 1 if pd.notna(r.get("predicted_pattern")) and r["predicted_pattern"] == r["ground_truth_pattern"]
                      else (0 if pd.notna(r.get("predicted_pattern")) else ""),
            axis=1,
        )

        # Per-pattern metrics
        all_patterns = sorted(df["ground_truth_pattern"].unique())
        per_pattern = []

        for p in all_patterns:
            gt_p   = df["ground_truth_pattern"] == p
            pred_p = df["predicted_pattern"] == p

            tp = int((gt_p & pred_p).sum())
            fp = int((~gt_p & pred_p).sum())
            fn = int((gt_p & ~pred_p).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1   = (2 * prec * rec / (prec + rec)
                    if (not math.isnan(prec) and not math.isnan(rec) and (prec + rec) > 0)
                    else float("nan"))

            per_pattern.append(PatternResult(
                pattern=p,
                gt_count=int(gt_p.sum()),
                pred_count=int(pred_p.sum()),
                tp=tp, fp=fp, fn=fn,
                precision=prec, recall=rec, f1=f1,
            ))

        total_tp   = sum(m.tp for m in per_pattern)
        total_pred = int(df["predicted_pattern"].notna().sum())
        total_gt   = len(df)
        total_miss = int(df["predicted_pattern"].isna().sum())

        micro_p  = total_tp / total_pred if total_pred > 0 else 0.0
        micro_r  = total_tp / total_gt   if total_gt  > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) > 0 else 0.0)

    # Observed macro: only average over patterns present in this framework's GT
    valid_prec = [m.precision for m in per_pattern if not math.isnan(m.precision)]
    valid_rec  = [m.recall    for m in per_pattern if not math.isnan(m.recall)]
    valid_f1   = [m.f1        for m in per_pattern if not math.isnan(m.f1)]

    obs_macro_p  = sum(valid_prec) / len(valid_prec) if valid_prec else 0.0
    obs_macro_r  = sum(valid_rec)  / len(valid_rec)  if valid_rec  else 0.0
    obs_macro_f1 = sum(valid_f1)   / len(valid_f1)   if valid_f1   else 0.0

    fw = FrameworkResult(
        name="",
        total_gt=total_gt,
        total_predicted=total_pred,
        total_tp=total_tp,
        total_missed=total_miss,
        multi_label=multi_label,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        observed_macro_recall=obs_macro_r,
        observed_macro_precision=obs_macro_p,
        observed_macro_f1=obs_macro_f1,
        per_pattern=per_pattern,
    )
    return df, per_pattern, fw


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_framework(name: str, fw: FrameworkResult) -> None:
    mode = "multi-label" if fw.multi_label else "single-label"
    print(f"\n{'=' * 80}")
    print(f"  Framework: {name}  [{mode}]")
    print(f"{'=' * 80}")
    print(f"  GT files             : {fw.total_gt}")
    print(f"  Files with prediction: {fw.total_predicted}  ({fw.total_predicted / fw.total_gt * 100:.0f} %)")
    print(f"  Files missed         : {fw.total_missed}")
    print(f"  Correct (TP)         : {fw.total_tp}")
    print()
    print(f"  Micro  P / R / F1       : {fw.micro_precision:.3f} / {fw.micro_recall:.3f} / {fw.micro_f1:.3f}")
    print(f"  Observed Macro P / R / F1: {fw.observed_macro_precision:.3f} / {fw.observed_macro_recall:.3f} / {fw.observed_macro_f1:.3f}")
    print()
    print(f"  {'Pattern':<48} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-' * 100}")
    for m in sorted(fw.per_pattern, key=lambda x: x.gt_count, reverse=True):
        print(
            f"  {m.pattern:<48} {m.gt_count:>4} {m.pred_count:>5} {m.tp:>4} {m.fp:>4} {m.fn:>4}"
            f" {_nan(m.precision):>6} {_nan(m.recall):>6} {_nan(m.f1):>6}"
        )
    print()


def write_markdown(name: str, fw: FrameworkResult, df: pd.DataFrame, report_path: Path) -> None:
    mode = "multi-label" if fw.multi_label else "single-label"
    lines: list[str] = []
    lines.append(f"# Precision / Recall Evaluation — {name}\n\n")
    lines.append(f"_Evaluation mode: **{mode}**_\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"| Metric | Value |\n|---|---|\n")
    lines.append(f"| GT files | {fw.total_gt} |\n")
    lines.append(f"| Files with prediction | {fw.total_predicted} ({fw.total_predicted / fw.total_gt * 100:.0f} %) |\n")
    lines.append(f"| Files missed | {fw.total_missed} |\n")
    lines.append(f"| Correct (TP) | {fw.total_tp} |\n")
    lines.append(f"| Micro Precision | {fw.micro_precision:.3f} |\n")
    lines.append(f"| Micro Recall | {fw.micro_recall:.3f} |\n")
    lines.append(f"| Micro F1 | {fw.micro_f1:.3f} |\n")
    lines.append(f"| **Observed Macro Recall** | **{fw.observed_macro_recall:.3f}** |\n")
    lines.append(f"| **Observed Macro Precision** | **{fw.observed_macro_precision:.3f}** |\n")
    lines.append(f"| **Observed Macro F1** | **{fw.observed_macro_f1:.3f}** |\n")

    lines.append("\n## Per-Pattern Results\n\n")
    if fw.multi_label:
        lines.append(
            "_Multi-label mode: TP = GT pattern detected in ANY prediction for the file. "
            "FP = pattern predicted for a file whose GT is different. "
            "Observed recall = TP / (TP + FN) over patterns present in GT._\n\n"
        )
    else:
        lines.append(
            "_Single-label mode: only the highest-scoring prediction per file is evaluated. "
            "Observed recall = TP / (TP + FN) over patterns present in GT._\n\n"
        )
    lines.append("| Pattern | GT | Pred | TP | FP | FN | Precision | Recall | F1 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for m in sorted(fw.per_pattern, key=lambda x: x.gt_count, reverse=True):
        lines.append(
            f"| {m.pattern} | {m.gt_count} | {m.pred_count} | {m.tp} | {m.fp} | {m.fn} "
            f"| {_nan(m.precision)} | {_nan(m.recall)} | {_nan(m.f1)} |\n"
        )

    lines.append("\n## File-Level Predictions\n\n")
    if fw.multi_label:
        lines.append("| File | GT Pattern | All Detected Patterns | Top Prediction | Match |\n")
        lines.append("|---|---|---|---|---|\n")
        for _, row in df.iterrows():
            fname = Path(row["file_path"]).name
            gt_p  = row["ground_truth_pattern"]
            top   = row.get("top_predicted_pattern", "—")
            if pd.isna(top):
                top = "—"
            all_pats = row.get("all_predicted_patterns")
            all_str = ", ".join(sorted(all_pats)) if isinstance(all_pats, set) else "—"
            match = "✓" if row.get("gt_detected") else ("✗" if isinstance(all_pats, set) else "—")
            lines.append(f"| `{fname}` | {gt_p} | {all_str} | {top} | {match} |\n")
    else:
        lines.append("| File | GT Pattern | Predicted Pattern | Score | Match |\n")
        lines.append("|---|---|---|---|---|\n")
        pred_col  = "predicted_pattern" if "predicted_pattern" in df.columns else None
        score_col = "pred_score" if "pred_score" in df.columns else None
        for _, row in df.iterrows():
            fname = Path(row["file_path"]).name
            gt_p  = row["ground_truth_pattern"]
            pred  = row[pred_col] if pred_col and pd.notna(row.get(pred_col)) else "—"
            score = f"{row[score_col]:.4f}" if score_col and pd.notna(row.get(score_col)) else "—"
            match = str(row.get("match", ""))
            lines.append(f"| `{fname}` | {gt_p} | {pred} | {score} | {match} |\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"  Report written to: {report_path}")


# ---------------------------------------------------------------------------
# Cross-framework comparison
# ---------------------------------------------------------------------------

def print_comparison(results: list[tuple[str, FrameworkResult]]) -> None:
    print(f"\n{'=' * 80}")
    print("  CROSS-FRAMEWORK COMPARISON")
    print(f"{'=' * 80}")
    header = (
        f"  {'Framework':<45} {'Mode':<12} {'Files':>6} {'Cov%':>5}"
        f" {'Micro-P':>8} {'Micro-R':>8} {'Micro-F1':>9}"
        f" {'Obs-R':>7} {'Obs-P':>7} {'Obs-F1':>8}"
    )
    print(header)
    print(f"  {'-' * 120}")
    for name, fw in results:
        cov  = fw.total_predicted / fw.total_gt * 100
        mode = "multi-label" if fw.multi_label else "single-label"
        print(
            f"  {name:<45} {mode:<12} {fw.total_gt:>6} {cov:>4.0f}%"
            f" {fw.micro_precision:>8.3f} {fw.micro_recall:>8.3f} {fw.micro_f1:>9.3f}"
            f" {fw.observed_macro_recall:>7.3f} {fw.observed_macro_precision:>7.3f} {fw.observed_macro_f1:>8.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    all_results: list[tuple[str, FrameworkResult]] = []

    for cfg in CONFIGS:
        name = cfg["name"]
        multi_label = cfg.get("multi_label", False)
        df, per_pattern, fw = join_and_evaluate(cfg["gt"], cfg["pred"], multi_label=multi_label)
        fw.name = name
        print_framework(name, fw)
        write_markdown(name, fw, df, cfg["report"])
        all_results.append((name, fw))

    print_comparison(all_results)


if __name__ == "__main__":
    main()

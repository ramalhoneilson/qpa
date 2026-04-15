"""
Precision, recall, and F1 computation for the pattern-finder evaluation.

Each row in the ground truth CSV represents one file with a single
ground_truth_pattern label. The evaluator maps analysis output predictions
(one per file, highest score wins) and computes per-pattern and aggregate
metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class PatternMetrics:
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
class AggregateMetrics:
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total_gt: int
    total_predicted: int
    total_tp: int
    total_missed: int


def compute_metrics(eval_df: pd.DataFrame) -> tuple[list[PatternMetrics], AggregateMetrics]:
    """
    Compute per-pattern and aggregate precision/recall/F1.

    Parameters
    ----------
    eval_df : DataFrame with columns:
        - ground_truth_pattern  : the correct label for the file
        - predicted_pattern     : the system's prediction (NaN if no prediction)

    Returns
    -------
    (per_pattern, aggregate)
    """
    all_patterns = sorted(eval_df["ground_truth_pattern"].unique())
    per_pattern: list[PatternMetrics] = []

    for p in all_patterns:
        gt_p   = eval_df["ground_truth_pattern"] == p
        pred_p = eval_df["predicted_pattern"] == p

        tp = int((gt_p & pred_p).sum())
        fp = int((~gt_p & pred_p).sum())
        fn = int((gt_p & ~pred_p).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = 2 * prec * rec / (prec + rec) if (not math.isnan(prec) and not math.isnan(rec) and (prec + rec) > 0) else float("nan")

        per_pattern.append(PatternMetrics(
            pattern=p,
            gt_count=int(gt_p.sum()),
            pred_count=int(pred_p.sum()),
            tp=tp, fp=fp, fn=fn,
            precision=prec, recall=rec, f1=f1,
        ))

    total_tp   = sum(m.tp for m in per_pattern)
    total_pred = int(eval_df["predicted_pattern"].notna().sum())
    total_gt   = len(eval_df)
    total_missed = int(eval_df["predicted_pattern"].isna().sum())

    micro_p  = total_tp / total_pred if total_pred > 0 else 0.0
    micro_r  = total_tp / total_gt   if total_gt  > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    valid_p  = [m.precision for m in per_pattern if not math.isnan(m.precision)]
    valid_r  = [m.recall    for m in per_pattern if not math.isnan(m.recall)]
    valid_f1 = [m.f1        for m in per_pattern if not math.isnan(m.f1)]

    macro_p  = sum(valid_p)  / len(valid_p)  if valid_p  else 0.0
    macro_r  = sum(valid_r)  / len(valid_r)  if valid_r  else 0.0
    macro_f1 = sum(valid_f1) / len(valid_f1) if valid_f1 else 0.0

    agg = AggregateMetrics(
        micro_precision=micro_p,  micro_recall=micro_r,  micro_f1=micro_f1,
        macro_precision=macro_p,  macro_recall=macro_r,  macro_f1=macro_f1,
        total_gt=total_gt,
        total_predicted=total_pred,
        total_tp=total_tp,
        total_missed=total_missed,
    )
    return per_pattern, agg


def join_predictions(
    ground_truth_path: str,
    analysis_output_path: str,
    analysis_sep: str = ";",
) -> pd.DataFrame:
    """
    Load ground truth CSV and analysis output CSV, join them on file_path,
    keeping the highest-scoring prediction per file (ties broken alphabetically
    by pattern name).  Returns the ground truth DataFrame with predicted_pattern
    and match columns filled.
    """
    gt   = pd.read_csv(ground_truth_path)
    pred = pd.read_csv(analysis_output_path, sep=analysis_sep)

    pred_in_gt = pred[pred["file_path"].isin(gt["file_path"])].copy()

    pred_best = (
        pred_in_gt
        .sort_values(["file_path", "similarity_score", "pattern"], ascending=[True, False, True])
        .drop_duplicates(subset=["file_path"], keep="first")
        [["file_path", "pattern", "similarity_score"]]
        .rename(columns={"pattern": "predicted_pattern", "similarity_score": "pred_score"})
    )

    # Drop stale predicted_pattern / match columns if they exist
    drop_cols = [c for c in ["predicted_pattern", "match", "pred_score"] if c in gt.columns]
    gt = gt.drop(columns=drop_cols)

    result = gt.merge(pred_best, on="file_path", how="left")

    result["match"] = result.apply(
        lambda r: 1 if (pd.notna(r["predicted_pattern"]) and r["predicted_pattern"] == r["ground_truth_pattern"])
                  else (0 if pd.notna(r["predicted_pattern"]) else ""),
        axis=1,
    )
    return result

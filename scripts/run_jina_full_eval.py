"""Full analysis + evaluation pipeline using jinaai/jina-embeddings-v2-base-code.

Completely self-contained — does NOT modify any project config or source files.
Outputs go to data/jina_* so baseline files are never overwritten.

Steps
-----
1. Run semantic analysis on braket and qiskit-ml eval dirs with Jina model.
2. Evaluate both outputs against ground truth (same logic as
   evaluate_braket_and_qiskit_ml.py).
3. Print side-by-side comparison with the all-mpnet-base-v2 baseline.
"""

from __future__ import annotations

import ast
import csv
import gc
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

JINA_MODEL_ID = "jinaai/jina-embeddings-v2-base-code"

SIMILARITY_THRESHOLDS = {"name": 0.90, "summary": 0.65}

CONCEPT_FILES = [
    DATA / "classiq_quantum_concepts.json",
    DATA / "pennylane_quantum_concepts.json",
    DATA / "qiskit_quantum_concepts.json",
    DATA / "qiskit_algorithms_quantum_concepts.json",
]
PATTERN_FILES = [
    DATA / "knowledge_base/enriched_classiq_quantum_patterns.csv",
    DATA / "knowledge_base/enriched_pennylane_quantum_patterns.csv",
    DATA / "knowledge_base/enriched_qiskit_quantum_patterns.csv",
    DATA / "knowledge_base/enriched_qiskit_algorithms_quantum_patterns.csv",
]

EVAL_CONFIGS = [
    {
        "name":       "Amazon Braket Algorithm Library",
        "target_dir": DATA / "eval_notebooks/braket_algorithm_library",
        "gt":         DATA / "braket_algorithm_library_ground_truth.csv",
        "out":        DATA / "jina_braket_eval_output.csv",
        "baseline":   DATA / "braket_algorithm_library_eval_output.csv",
        "multi_label": False,
    },
    {
        "name":       "Qiskit Machine Learning",
        "target_dir": DATA / "eval_notebooks/qiskit_machine_learning",
        "gt":         DATA / "qiskit_machine_learning_ground_truth.csv",
        "out":        DATA / "jina_qiskit_ml_eval_output.csv",
        "baseline":   DATA / "qiskit_machine_learning_eval_output.csv",
        "multi_label": True,
    },
]


# ── Analysis helpers (mirror of run_analysis.py, standalone) ───────────────────

def normalize_identifier(name: str) -> str:
    name = name.replace("_", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    return name.lower().strip()


def extract_short_name(full_name: str) -> str:
    if not full_name:
        return ""
    return full_name.split("/")[-1].split(".")[-1]


def load_patterns_map(file_paths: list[Path]) -> dict[str, str]:
    pattern_map = {}
    for path in file_paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3 and row[0].strip() and row[2].strip():
                    pattern_map[row[0].strip()] = row[2].strip()
    return pattern_map


def load_quantum_concepts(file_paths: list[Path], pattern_map: dict[str, str]) -> list[dict]:
    concepts = []
    short_map = {extract_short_name(k): v for k, v in pattern_map.items()}
    for path in file_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            if "name" not in item or "summary" not in item:
                continue
            full  = item["name"]
            short = extract_short_name(full)
            pattern = "N/A"
            if full in pattern_map:
                pattern = pattern_map[full]
            elif short in short_map:
                pattern = short_map[short]
            else:
                for k, v in pattern_map.items():
                    if full.endswith(k):
                        pattern = v
                        break
            concepts.append({"name": full, "summary": item["summary"],
                              "short_name": short, "pattern": pattern})
    return concepts


def extract_comments(file_path: Path) -> str:
    comments = []
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("#"):
                    t = s.lstrip("# ").strip()
                    if t:
                        comments.append(t)
    except Exception:
        pass
    return " ".join(comments)


def run_analysis(target_dir: Path, output_csv: Path,
                 model: SentenceTransformer,
                 quantum_concepts: list[dict],
                 concept_name_embeddings: np.ndarray,
                 concept_summary_embeddings: np.ndarray) -> None:
    script_files = list(target_dir.rglob("*.py"))
    total = len(script_files)
    print(f"  Scanning {total} files in '{target_dir.name}'...")

    best_matches: dict[tuple[str, str], dict] = {}

    for i, file_path in enumerate(script_files):
        if (i + 1) == total or (i + 1) % 5 == 0:
            print(f"    {i+1}/{total}")
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = str(file_path.relative_to(target_dir))

        # Name matching
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None

        located: list[tuple[str, int]] = []
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = None
                    if isinstance(func, ast.Name):
                        name = func.id
                    elif isinstance(func, ast.Attribute):
                        name = func.attr
                    if name:
                        located.append((name, node.lineno))

        if located:
            elems = [e[0] for e in located]
            # convert_to_numpy avoids keeping PyTorch tensors alive
            code_emb = model.encode(
                [normalize_identifier(n) for n in elems],
                convert_to_numpy=True, batch_size=16)
            sim = 1 - cdist(code_emb, concept_name_embeddings, "cosine")
            for ei, (elem, lineno) in enumerate(located):
                for ci, concept in enumerate(quantum_concepts):
                    score = float(sim[ei, ci])
                    if score >= SIMILARITY_THRESHOLDS["name"]:
                        key = (rel, concept["name"])
                        if key not in best_matches or score > best_matches[key]["score"]:
                            best_matches[key] = {
                                "file_path": rel, "concept_name": concept["name"],
                                "pattern": concept["pattern"], "match_type": "name",
                                "matched_text": elem, "score": score, "first_line": lineno,
                            }
            del code_emb, sim
            gc.collect()

        # Summary matching
        comment = extract_comments(file_path)
        if comment:
            c_emb = model.encode([comment], convert_to_numpy=True, batch_size=1)
            sim_s = 1 - cdist(c_emb, concept_summary_embeddings, "cosine")
            trunc = (comment[:150] + "...") if len(comment) > 150 else comment
            for ci, concept in enumerate(quantum_concepts):
                score = float(sim_s[0, ci])
                if score >= SIMILARITY_THRESHOLDS["summary"]:
                    key = (rel, concept["name"])
                    if key not in best_matches or score > best_matches[key]["score"]:
                        best_matches[key] = {
                            "file_path": rel, "concept_name": concept["name"],
                            "pattern": concept["pattern"], "match_type": "summary",
                            "matched_text": trunc.replace(";", ","),
                            "score": score, "first_line": 0,
                        }
            del c_emb, sim_s
            gc.collect()

    sorted_rows = sorted(best_matches.values(),
                         key=lambda r: (r["file_path"], r["first_line"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["file_path","concept_name","pattern","match_type",
                         "matched_text","similarity_score","first_line"])
        for row in sorted_rows:
            writer.writerow([row["file_path"], row["concept_name"], row["pattern"],
                             row["match_type"], row["matched_text"],
                             f"{row['score']:.4f}", row["first_line"]])
    print(f"  Saved {len(best_matches)} pairs → {output_csv.name}")


# ── Evaluation helpers (mirror of evaluate_braket_and_qiskit_ml.py) ────────────

def _nan(v: float) -> str:
    return f"{v:.3f}" if not math.isnan(v) else "  —  "


@dataclass
class PatternResult:
    pattern: str
    gt: int = 0; pred: int = 0; tp: int = 0; fp: int = 0; fn: int = 0
    precision: float = float("nan")
    recall:    float = float("nan")
    f1:        float = float("nan")


@dataclass
class FrameworkResult:
    name: str
    coverage: float
    micro_p: float; micro_r: float; micro_f1: float
    obs_r: float;   obs_p: float;   obs_f1: float
    per_pattern: list[PatternResult] = field(default_factory=list)


def evaluate(gt_path: Path, pred_path: Path, multi_label: bool) -> FrameworkResult:
    gt   = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path, sep=";")
    drop = [c for c in ["predicted_pattern","pred_score","match",
                        "all_predicted_patterns","gt_detected"] if c in gt.columns]
    gt = gt.drop(columns=drop)

    pred_in_gt = pred[pred["file_path"].isin(gt["file_path"])].copy()

    if multi_label:
        all_preds = (pred_in_gt.groupby("file_path")["pattern"]
                     .apply(lambda s: set(s.tolist()))
                     .reset_index()
                     .rename(columns={"pattern": "all_predicted_patterns"}))
        df = gt.merge(all_preds, on="file_path", how="left")
        df["gt_detected"] = df.apply(
            lambda r: isinstance(r.get("all_predicted_patterns"), set)
                      and r["ground_truth_pattern"] in r["all_predicted_patterns"], axis=1)

        patterns = sorted(df["ground_truth_pattern"].unique())
        per = []
        for p in patterns:
            gp = df["ground_truth_pattern"] == p
            tp = int((gp & df["gt_detected"]).sum())
            fn = int((gp & ~df["gt_detected"]).sum())
            fp = int((~gp & df["all_predicted_patterns"].apply(
                lambda s: p in s if isinstance(s, set) else False)).sum())
            pc = int(df["all_predicted_patterns"].apply(
                lambda s: p in s if isinstance(s, set) else False).sum())
            prec = tp/(tp+fp) if (tp+fp) > 0 else float("nan")
            rec  = tp/(tp+fn) if (tp+fn) > 0 else float("nan")
            f1   = (2*prec*rec/(prec+rec)
                    if not math.isnan(prec) and not math.isnan(rec) and (prec+rec)>0
                    else float("nan"))
            per.append(PatternResult(p, int(gp.sum()), pc, tp, fp, fn, prec, rec, f1))

        total_tp   = sum(m.tp for m in per)
        total_pred = int(df["all_predicted_patterns"].notna().sum())
        total_gt   = len(df)
        micro_p  = total_tp/total_pred if total_pred > 0 else 0.0
        micro_r  = total_tp/total_gt   if total_gt  > 0 else 0.0
        micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r)>0 else 0.0
        cov = total_pred/total_gt if total_gt > 0 else 0.0

    else:
        pred_best = (pred_in_gt
                     .sort_values(["file_path","similarity_score","pattern"],
                                  ascending=[True,False,True])
                     .drop_duplicates("file_path", keep="first")
                     [["file_path","pattern","similarity_score"]]
                     .rename(columns={"pattern":"predicted_pattern",
                                      "similarity_score":"pred_score"}))
        df = gt.merge(pred_best, on="file_path", how="left")
        df["gt_detected"] = df["predicted_pattern"] == df["ground_truth_pattern"]

        patterns = sorted(df["ground_truth_pattern"].unique())
        per = []
        for p in patterns:
            gp   = df["ground_truth_pattern"] == p
            pp   = df["predicted_pattern"] == p
            tp   = int((gp & pp).sum())
            fp   = int((~gp & pp).sum())
            fn   = int((gp & ~pp).sum())
            prec = tp/(tp+fp) if (tp+fp)>0 else float("nan")
            rec  = tp/(tp+fn) if (tp+fn)>0 else float("nan")
            f1   = (2*prec*rec/(prec+rec)
                    if not math.isnan(prec) and not math.isnan(rec) and (prec+rec)>0
                    else float("nan"))
            per.append(PatternResult(p, int(gp.sum()), int(pp.sum()), tp, fp, fn, prec, rec, f1))

        total_tp   = df["gt_detected"].sum()
        total_pred = df["predicted_pattern"].notna().sum()
        total_gt   = len(df)
        micro_p  = total_tp/total_pred if total_pred > 0 else 0.0
        micro_r  = total_tp/total_gt   if total_gt  > 0 else 0.0
        micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r)>0 else 0.0
        cov = total_pred/total_gt if total_gt > 0 else 0.0

    obs_patterns = [m for m in per if m.gt > 0 and not math.isnan(m.recall)]
    obs_r  = sum(m.recall    for m in obs_patterns)/len(obs_patterns) if obs_patterns else 0.0
    obs_p  = sum(m.precision for m in obs_patterns
                 if not math.isnan(m.precision))/len(obs_patterns) if obs_patterns else 0.0
    obs_f1 = (2*obs_p*obs_r/(obs_p+obs_r)) if (obs_p+obs_r)>0 else 0.0

    name = gt_path.stem.replace("_ground_truth","").replace("_"," ").title()
    return FrameworkResult(name, cov, micro_p, micro_r, micro_f1,
                           obs_r, obs_p, obs_f1, per)


def print_framework(fw: FrameworkResult, multi_label: bool) -> None:
    mode = "multi-label" if multi_label else "single-label"
    print(f"\n  {'='*60}")
    print(f"  {fw.name}  [{mode}]")
    print(f"  {'='*60}")
    print(f"  Coverage  : {fw.coverage:.0%}")
    print(f"  Micro  P/R/F1      : {fw.micro_p:.3f} / {fw.micro_r:.3f} / {fw.micro_f1:.3f}")
    print(f"  Observed P/R/F1    : {fw.obs_p:.3f} / {fw.obs_r:.3f} / {fw.obs_f1:.3f}")
    print()
    print(f"  {'Pattern':50s} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*95}")
    for p in fw.per_pattern:
        print(f"  {p.pattern:50s} {p.gt:>4} {p.pred:>5} {p.tp:>4} {p.fp:>4} {p.fn:>4}"
              f"  {_nan(p.precision):>6}  {_nan(p.recall):>6}  {_nan(p.f1):>6}")


def print_comparison(baseline: FrameworkResult, jina: FrameworkResult,
                     multi_label: bool) -> None:
    mode = "multi-label" if multi_label else "single-label"
    print(f"\n  ── {jina.name} [{mode}] ──")
    print(f"  {'Metric':20s}  {'Baseline (mpnet)':>18}  {'Jina v2':>10}  {'Δ':>8}")
    print(f"  {'-'*60}")

    def row(label, b, j):
        delta = j - b
        sign = "+" if delta >= 0 else ""
        print(f"  {label:20s}  {b:>18.3f}  {j:>10.3f}  {sign}{delta:>7.3f}")

    row("Coverage",   baseline.coverage,  jina.coverage)
    row("Micro-P",    baseline.micro_p,   jina.micro_p)
    row("Micro-R",    baseline.micro_r,   jina.micro_r)
    row("Micro-F1",   baseline.micro_f1,  jina.micro_f1)
    row("Obs-Recall", baseline.obs_r,     jina.obs_r)
    row("Obs-Precis", baseline.obs_p,     jina.obs_p)
    row("Obs-F1",     baseline.obs_f1,    jina.obs_f1)

    # Per-pattern recall comparison
    b_map = {p.pattern: p for p in baseline.per_pattern}
    j_map = {p.pattern: p for p in jina.per_pattern}
    all_patterns = sorted(set(b_map) | set(j_map))
    print(f"\n  {'Pattern':50s}  {'Base Rec':>9}  {'Jina Rec':>9}  {'Δ':>7}")
    print(f"  {'-'*80}")
    for pat in all_patterns:
        br = b_map[pat].recall if pat in b_map else float("nan")
        jr = j_map[pat].recall if pat in j_map else float("nan")
        if math.isnan(br) and math.isnan(jr):
            continue
        delta = (jr - br) if not math.isnan(br) and not math.isnan(jr) else float("nan")
        sign = ("+" if delta >= 0 else "") if not math.isnan(delta) else ""
        print(f"  {pat:50s}  {_nan(br):>9}  {_nan(jr):>9}  {sign}{_nan(delta):>7}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  PHASE 1: Load model + KB")
    print("=" * 65)

    print(f"\nLoading Jina model: {JINA_MODEL_ID}")
    model = SentenceTransformer(JINA_MODEL_ID, trust_remote_code=True)
    model.eval()

    pattern_map     = load_patterns_map(PATTERN_FILES)
    quantum_concepts = load_quantum_concepts(CONCEPT_FILES, pattern_map)
    print(f"KB: {len(quantum_concepts)} concepts, {len(pattern_map)} pattern mappings")

    short_names = [c["short_name"] for c in quantum_concepts]
    summaries   = [c["summary"]    for c in quantum_concepts]

    print("Encoding KB name embeddings...")
    concept_name_emb = model.encode(
        [normalize_identifier(n) for n in short_names],
        convert_to_numpy=True, batch_size=32)
    gc.collect()
    print("Encoding KB summary embeddings...")
    concept_summary_emb = model.encode(
        summaries, convert_to_numpy=True, batch_size=16)
    gc.collect()

    print("=" * 65)
    print("  PHASE 2: Run analysis on eval dirs")
    print("=" * 65)

    jina_results: list[FrameworkResult] = []
    base_results: list[FrameworkResult] = []

    for cfg in EVAL_CONFIGS:
        print(f"\n--- {cfg['name']} ---")
        run_analysis(
            target_dir=cfg["target_dir"],
            output_csv=cfg["out"],
            model=model,
            quantum_concepts=quantum_concepts,
            concept_name_embeddings=concept_name_emb,
            concept_summary_embeddings=concept_summary_emb,
        )

    print("\n" + "=" * 65)
    print("  PHASE 3: Evaluate Jina output")
    print("=" * 65)

    for cfg in EVAL_CONFIGS:
        fw = evaluate(cfg["gt"], cfg["out"], cfg["multi_label"])
        print_framework(fw, cfg["multi_label"])
        jina_results.append(fw)

    print("\n" + "=" * 65)
    print("  PHASE 4: Load baseline and compare")
    print("=" * 65)

    for cfg in EVAL_CONFIGS:
        fw_base = evaluate(cfg["gt"], cfg["baseline"], cfg["multi_label"])
        base_results.append(fw_base)

    print("\n\n" + "=" * 65)
    print("  COMPARISON: Baseline (all-mpnet-base-v2) vs Jina v2")
    print("=" * 65)

    for base, jina, cfg in zip(base_results, jina_results, EVAL_CONFIGS):
        print_comparison(base, jina, cfg["multi_label"])

    print("\n\nDone.")


if __name__ == "__main__":
    main()

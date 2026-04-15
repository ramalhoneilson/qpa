"""Standalone model comparison: all-mpnet-base-v2 vs jina-embeddings-v2-base-code.

Uses real data from the project (KB concepts + matched code elements) but
does NOT import or modify any project modules or config.

Outputs:
  - Similarity score distributions for TP pairs (known matches) and
    TN pairs (known non-matches)
  - Suggested threshold for each model to match the current TP rate
  - Encoding speed comparison
  - scripts/model_comparison_report.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

# ── Paths (relative to project root — run from there) ─────────────────────────
DATA = Path("data")
KB_FILES = [
    DATA / "classiq_quantum_concepts.json",
    DATA / "pennylane_quantum_concepts.json",
    DATA / "qiskit_quantum_concepts.json",
    DATA / "qiskit_algorithms_quantum_concepts.json",
]
BASELINE_CSV = DATA / "quantum_concept_matches_with_patterns.csv"
REPORT_PATH  = Path("scripts/model_comparison_report.md")

MODELS = {
    "all-mpnet-base-v2":           "sentence-transformers/all-mpnet-base-v2",
    "jina-embeddings-v2-base-code": "jinaai/jina-embeddings-v2-base-code",
}

# Current production thresholds
CURRENT_NAME_THRESH    = 0.90
CURRENT_SUMMARY_THRESH = 0.65


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """Same normalisation as run_analysis.py."""
    import re
    name = name.replace("_", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    return name.lower().strip()


def short_name(full: str) -> str:
    return full.split("/")[-1].split(".")[-1]


def load_kb() -> tuple[list[str], list[str], list[str]]:
    """Returns (short_names, summaries, patterns)."""
    shorts, summaries, patterns_list = [], [], []
    for p in KB_FILES:
        data = json.loads(p.read_text(encoding="utf-8"))
        for c in data:
            shorts.append(short_name(c["name"]))
            summaries.append(c["summary"])
            patterns_list.append(c.get("pattern", "N/A"))
    return shorts, summaries, patterns_list


def load_baseline() -> tuple[list[str], list[str]]:
    """Returns (name_match_elements, summary_match_comments) from production output."""
    df = pd.read_csv(BASELINE_CSV, sep=";", dtype=str)
    name_elems    = df[df["match_type"] == "name"]["matched_text"].dropna().unique().tolist()
    summary_texts = df[df["match_type"] == "summary"]["matched_text"].dropna().unique().tolist()
    return name_elems, summary_texts


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1 - cdist(a, b, "cosine")


def percentile_summary(arr: np.ndarray, label: str) -> str:
    return (
        f"{label}: "
        f"min={arr.min():.3f}  "
        f"p25={np.percentile(arr,25):.3f}  "
        f"p50={np.percentile(arr,50):.3f}  "
        f"p75={np.percentile(arr,75):.3f}  "
        f"p90={np.percentile(arr,90):.3f}  "
        f"max={arr.max():.3f}"
    )


def threshold_stats(sim_matrix: np.ndarray, thresh: float) -> dict:
    """For each row (query), the best match score and whether it clears thresh."""
    best = sim_matrix.max(axis=1)
    fired = (best >= thresh).sum()
    return {
        "thresh": thresh,
        "fired": int(fired),
        "total": len(best),
        "fire_rate": fired / len(best),
        "mean_best": float(best.mean()),
        "p50_best": float(np.median(best)),
        "p90_best": float(np.percentile(best, 90)),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_model(label: str, model_id: str,
              concept_shorts: list[str],
              concept_summaries: list[str],
              name_elements: list[str],
              summary_texts: list[str]) -> dict:

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    model = SentenceTransformer(model_id, trust_remote_code=True)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    results = {"label": label, "model_id": model_id, "load_time": load_time}

    # ── Name channel ────────────────────────────────────────────────────────
    print(f"  Encoding {len(concept_shorts)} KB short names + {len(name_elements)} code elements...")
    t1 = time.perf_counter()
    kb_name_emb  = model.encode([normalize(n) for n in concept_shorts], convert_to_tensor=True)
    code_emb     = model.encode([normalize(n) for n in name_elements],  convert_to_tensor=True)
    name_enc_time = time.perf_counter() - t1

    sim_names = cosine_sim_matrix(code_emb.cpu().numpy(), kb_name_emb.cpu().numpy())
    # TP pairs: each code element's best matching KB concept
    tp_name_scores = sim_names.max(axis=1)

    results["name_enc_time"]  = name_enc_time
    results["name_tp_scores"] = tp_name_scores

    print(f"  Name encoding: {name_enc_time:.2f}s")
    print(f"  {percentile_summary(tp_name_scores, 'Name TP scores')}")

    for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]:
        s = threshold_stats(sim_names, t)
        marker = " ◄ current" if abs(t - CURRENT_NAME_THRESH) < 0.001 else ""
        print(f"    thresh={t:.2f}  fire_rate={s['fire_rate']:.2%}  ({s['fired']}/{s['total']} elements){marker}")

    # ── Summary channel ──────────────────────────────────────────────────────
    print(f"\n  Encoding {len(concept_summaries)} KB summaries + {len(summary_texts)} comment blocks...")
    t2 = time.perf_counter()
    kb_sum_emb   = model.encode(concept_summaries,  convert_to_tensor=True)
    comment_emb  = model.encode(summary_texts[:50], convert_to_tensor=True)  # cap at 50 for speed
    sum_enc_time = time.perf_counter() - t2

    sim_summaries = cosine_sim_matrix(comment_emb.cpu().numpy(), kb_sum_emb.cpu().numpy())
    tp_sum_scores = sim_summaries.max(axis=1)

    results["sum_enc_time"]  = sum_enc_time
    results["sum_tp_scores"] = tp_sum_scores

    print(f"  Summary encoding: {sum_enc_time:.2f}s")
    print(f"  {percentile_summary(tp_sum_scores, 'Summary TP scores')}")

    for t in [0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75]:
        s = threshold_stats(sim_summaries, t)
        marker = " ◄ current" if abs(t - CURRENT_SUMMARY_THRESH) < 0.001 else ""
        print(f"    thresh={t:.2f}  fire_rate={s['fire_rate']:.2%}  ({s['fired']}/{s['total']} files){marker}")

    # ── Spot-check known pairs ────────────────────────────────────────────────
    spot_pairs = [
        ("QFT",          "qft",          "exact match (QFT)"),
        ("grovers_search","grover_search","near match (Grover)"),
        ("QAOA",         "QAOA",         "exact match (QAOA)"),
        ("VQE",          "VQE",          "exact match (VQE)"),
        ("bernstein_vazirani_circuit", "phase_oracle", "gap case (BV→Oracle)"),
        ("bell_singlet", "prepare_bell_state",         "gap case (Bell→Entanglement)"),
        ("random_circuit","random_circuit","utility exact"),
        ("fit",          "VQE",          "noise (fit→VQE)"),
        ("append",       "qft",          "noise (append→QFT)"),
    ]
    print("\n  Spot-check pairs:")
    spot_results = []
    for elem, concept, desc in spot_pairs:
        e_emb = model.encode([normalize(elem)],    convert_to_tensor=True).cpu().numpy()
        c_emb = model.encode([normalize(concept)], convert_to_tensor=True).cpu().numpy()
        score = float(1 - cdist(e_emb, c_emb, "cosine")[0, 0])
        fires_name = "✓" if score >= CURRENT_NAME_THRESH else "✗"
        print(f"    {fires_name} {desc:45s}  {score:.4f}")
        spot_results.append((desc, score))
    results["spot"] = spot_results

    return results


def write_report(all_results: list[dict]) -> None:
    lines = [
        "# Embedding Model Comparison Report",
        "",
        f"Models compared: {', '.join(r['label'] for r in all_results)}",
        f"KB concepts: {sum(1 for _ in open(KB_FILES[0]))} (classiq only — see script for full count)",
        "",
        "## Load Time",
        "",
        "| Model | Load (s) |",
        "|---|---|",
    ]
    for r in all_results:
        lines.append(f"| {r['label']} | {r['load_time']:.1f} |")

    lines += ["", "## Name Channel — Best-match score distribution (code element → KB concept)", ""]
    for r in all_results:
        s = r["name_tp_scores"]
        lines.append(f"**{r['label']}**  "
                     f"p50={np.median(s):.3f}  p75={np.percentile(s,75):.3f}  "
                     f"p90={np.percentile(s,90):.3f}  max={s.max():.3f}  "
                     f"enc={r['name_enc_time']:.2f}s")
        lines.append("")

    lines += ["## Summary Channel — Best-match score distribution (comment block → KB summary)", ""]
    for r in all_results:
        s = r["sum_tp_scores"]
        lines.append(f"**{r['label']}**  "
                     f"p50={np.median(s):.3f}  p75={np.percentile(s,75):.3f}  "
                     f"p90={np.percentile(s,90):.3f}  max={s.max():.3f}  "
                     f"enc={r['sum_enc_time']:.2f}s")
        lines.append("")

    lines += ["## Spot-check Pairs", ""]
    header = "| Pair | " + " | ".join(r["label"] for r in all_results) + " |"
    sep    = "|---|" + "---|" * len(all_results)
    lines += [header, sep]
    for i, (desc, _) in enumerate(all_results[0]["spot"]):
        scores = " | ".join(f"{r['spot'][i][1]:.4f}" for r in all_results)
        lines.append(f"| {desc} | {scores} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


def main():
    print("Loading KB and baseline data...")
    concept_shorts, concept_summaries, _ = load_kb()
    name_elements, summary_texts = load_baseline()
    print(f"KB concepts: {len(concept_shorts)}")
    print(f"Code elements (name matches): {len(name_elements)}")
    print(f"Comment blocks (summary matches): {len(summary_texts)}")

    all_results = []
    for label, model_id in MODELS.items():
        result = run_model(
            label, model_id,
            concept_shorts, concept_summaries,
            name_elements, summary_texts,
        )
        all_results.append(result)

    write_report(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()

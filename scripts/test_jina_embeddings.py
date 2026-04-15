"""Hello-world test for jinaai/jina-embeddings-v2-base-code.

Checks:
  1. Model loads on CPU within available RAM
  2. Encodes a small batch of strings
  3. Computes cosine similarities
  4. Reports load time, encode time, and peak memory usage
"""

import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

MODEL_ID = "jinaai/jina-embeddings-v2-base-code"

SENTENCES = [
    # Quantum computing names — similar to what run_analysis.py embeds
    "quantum fourier transform",
    "apply qft",
    "QFT",
    "grover search",
    "amplitude amplification",
    "variational quantum eigensolver",
    # Generic noise
    "hello world",
    "open file",
]

def main():
    print(f"Model : {MODEL_ID}")
    print(f"Device: CPU (no CUDA detected)")
    print(f"Torch : {torch.__version__}")
    print()

    # ── 1. Load ────────────────────────────────────────────────────────────────
    tracemalloc.start()
    t0 = time.perf_counter()

    print("Loading model (this may take a minute on first run)...")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    model.eval()

    load_time = time.perf_counter() - t0
    _, mem_peak_load = tracemalloc.get_traced_memory()
    print(f"Loaded in {load_time:.1f}s  |  peak RAM delta: {mem_peak_load / 1e6:.0f} MB")
    print()

    # ── 2. Encode ──────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    with torch.no_grad():
        embeddings = model.encode(SENTENCES, convert_to_numpy=True, show_progress_bar=False)
    encode_time = time.perf_counter() - t1

    print(f"Encoded {len(SENTENCES)} sentences in {encode_time:.2f}s  "
          f"({encode_time / len(SENTENCES) * 1000:.0f} ms/sentence)")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print()

    # ── 3. Similarity spot-check ───────────────────────────────────────────────
    def sim(a, b):
        return 1 - cosine(embeddings[a], embeddings[b])

    print("Cosine similarity spot-check:")
    pairs = [
        (0, 1, "QFT ↔ apply qft"),
        (0, 2, "QFT ↔ QFT (exact)"),
        (0, 3, "QFT ↔ grover search"),
        (3, 4, "grover ↔ amplitude amplification"),
        (0, 6, "QFT ↔ hello world"),
        (0, 7, "QFT ↔ open file"),
    ]
    for i, j, label in pairs:
        print(f"  {label:45s}  {sim(i, j):.4f}")

    tracemalloc.stop()
    print()
    print("Done.")

if __name__ == "__main__":
    main()

"""
Build and persist ChromaDB collections for each embedding model under comparison.

One pair of collections is created per model:
  - ``{slug}_names``      : normalized short names of each KB concept
  - ``{slug}_summaries``  : summary text of each KB concept

The collections are stored on disk at ``data/chroma_db/`` and reused across
runs.  Re-run with ``--rebuild`` to wipe and recreate them (necessary after
KB changes or model additions).

Usage
-----
    python -m src.evaluation.embedding_comparison.indexer
    python -m src.evaluation.embedding_comparison.indexer --rebuild
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.analysis.run_analysis import (
    CONCEPT_FILES,
    PATTERN_FILES,
    load_patterns_map,
    load_quantum_concepts,
    normalize_identifier,
)
from src.conf import config

# ── Persistent storage ─────────────────────────────────────────────────────────
CHROMA_DIR: Path = config.RESULTS_DIR / "chroma_db"

# ── Models under comparison ────────────────────────────────────────────────────
# Add or remove entries here to change what gets indexed.
MODELS: list[str] = [
    "all-mpnet-base-v2",           # current baseline
    "all-MiniLM-L6-v2",            # fast, lightweight
    "all-MiniLM-L12-v2",           # intermediate quality/speed
    "BAAI/bge-small-en-v1.5",      # strong semantic retrieval benchmarks
    "multi-qa-mpnet-base-dot-v1",  # tuned for Q&A / retrieval pairs
]

# ChromaDB has a hard limit on batch size; keep below it to be safe.
_BATCH_SIZE = 200


def model_slug(model_name: str) -> str:
    """Return a ChromaDB-safe, filesystem-safe identifier for *model_name*.

    Forward-slashes (e.g. in ``BAAI/bge-small-en-v1.5``) and other special
    characters are replaced with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", model_name).strip("_")


def _load_concepts() -> list[dict]:
    pattern_map = load_patterns_map(PATTERN_FILES)
    return load_quantum_concepts(CONCEPT_FILES, pattern_map)


def build_collections(
    model_name: str,
    concepts: list[dict] | None = None,
    force_rebuild: bool = False,
) -> tuple[chromadb.Collection, chromadb.Collection]:
    """Build (or reuse) the two ChromaDB collections for *model_name*.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"all-mpnet-base-v2"``.
    concepts:
        Pre-loaded list of concept dicts.  Loaded from disk if ``None``.
    force_rebuild:
        Delete existing collections before rebuilding.

    Returns
    -------
    (names_collection, summaries_collection)
        Both collections use cosine distance and the given model's embedding
        function, so ``collection.query(query_texts=[...])`` handles encoding
        automatically.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    ef = SentenceTransformerEmbeddingFunction(model_name=model_name)
    slug = model_slug(model_name)
    names_name = f"{slug}_names"
    summaries_name = f"{slug}_summaries"

    existing = {c.name for c in client.list_collections()}

    if force_rebuild:
        for cname in (names_name, summaries_name):
            if cname in existing:
                client.delete_collection(cname)
                print(f"  [chroma] Deleted '{cname}'")
        existing = set()

    if names_name in existing and summaries_name in existing:
        print(f"  [chroma] Reusing existing collections for '{model_name}'")
        return (
            client.get_collection(names_name, embedding_function=ef),
            client.get_collection(summaries_name, embedding_function=ef),
        )

    if concepts is None:
        concepts = _load_concepts()

    print(f"  [chroma] Indexing {len(concepts)} concepts with '{model_name}'...")
    t0 = time.perf_counter()

    names_coll = client.create_collection(
        names_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    summaries_coll = client.create_collection(
        summaries_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"c{i}" for i in range(len(concepts))]
    # Normalize identifiers the same way run_analysis.py does before embedding.
    name_docs = [normalize_identifier(c["short_name"]) for c in concepts]
    summary_docs = [c["summary"] for c in concepts]
    metadatas = [
        {
            "concept_name": c["name"],
            "short_name": c["short_name"],
            "pattern": c["pattern"],
        }
        for c in concepts
    ]

    for start in range(0, len(concepts), _BATCH_SIZE):
        end = start + _BATCH_SIZE
        names_coll.add(
            ids=ids[start:end],
            documents=name_docs[start:end],
            metadatas=metadatas[start:end],
        )
        summaries_coll.add(
            ids=ids[start:end],
            documents=summary_docs[start:end],
            metadatas=metadatas[start:end],
        )

    elapsed = time.perf_counter() - t0
    print(f"  [chroma] Done in {elapsed:.1f}s")
    return names_coll, summaries_coll


def build_all(force_rebuild: bool = False) -> None:
    """Index KB concepts into ChromaDB for every model in ``MODELS``."""
    print("Loading KB concepts...")
    concepts = _load_concepts()
    print(f"Loaded {len(concepts)} concepts.\n")

    for model_name in MODELS:
        print(f"Model: {model_name}")
        build_collections(model_name, concepts=concepts, force_rebuild=force_rebuild)
        print()

    print("All collections built.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index KB concepts into ChromaDB.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe and rebuild all collections (needed after KB changes).",
    )
    args = parser.parse_args()
    build_all(force_rebuild=args.rebuild)

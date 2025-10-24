import logging
import re
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer, util

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.95


def _to_snake_case(name: str) -> str:
    """Converts a PascalCase string to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _is_deprecated(concept: Dict[str, Any]) -> bool:
    """Checks if a concept is marked as deprecated in its source or docstring."""
    source = concept.get("source_code", "")
    docstring = concept.get("docstring", "")
    return "@deprecate" in source or "deprecated" in docstring.lower()


def deduplicate_by_naming_convention(
    concepts_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Removes functions that appear to be simple snake_case wrappers for a
    PascalCase class in the same module.
    """
    logging.info("--- Filtering: Deduplicating concepts by naming convention ---")
    concepts_by_module: Dict[str, List[Dict[str, Any]]] = {}
    for concept in concepts_data:
        module_path = ".".join(concept["name"].split(".")[:-1]).replace("/qiskit/", "")
        concepts_by_module.setdefault(module_path, []).append(concept)

    discard_full_names = set()
    for concepts_in_module in concepts_by_module.values():
        classes = {
            c["name"].split(".")[-1]: c
            for c in concepts_in_module
            if c["type"] == "Class"
        }
        functions = {
            f["name"].split(".")[-1]: f
            for f in concepts_in_module
            if f["type"] == "Function"
        }
        if not (classes and functions):
            continue

        for class_name, class_concept in classes.items():
            expected_func_name = _to_snake_case(class_name)
            if expected_func_name in functions:
                func_to_discard = functions[expected_func_name]
                discard_full_names.add(func_to_discard["name"])
                logging.info(
                    f"  -> Discarding wrapper function: {func_to_discard['name']} for class {class_concept['name']}"
                )

    final_concepts = [c for c in concepts_data if c["name"] not in discard_full_names]
    logging.info(
        f"Reduced concepts from {len(concepts_data)} to {len(final_concepts)}."
    )
    return final_concepts


def deduplicate_by_semantic_similarity(
    concepts_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Clusters concepts by semantic similarity of their summaries and keeps only
    the "best" concept from each cluster.
    """
    if not concepts_data:
        return []
    logging.info(
        f"--- Filtering: Deduplicating concepts by semantic similarity (threshold: {SIMILARITY_THRESHOLD}) ---"
    )
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logging.error(
            f"Could not load SentenceTransformer model '{EMBEDDING_MODEL_NAME}'. Please ensure it is installed. Error: {e}"
        )
        return concepts_data

    embeddings = model.encode(
        [c["summary"] for c in concepts_data],
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    clusters = util.community_detection(
        embeddings, min_community_size=1, threshold=SIMILARITY_THRESHOLD
    )

    final_concepts = []
    processed_indices = set()

    for cluster_indices in clusters:
        if not cluster_indices:
            continue

        cluster = [concepts_data[i] for i in cluster_indices]

        # Define the key for selecting the best concept in a cluster
        best_concept = max(
            cluster,
            key=lambda c: (
                not _is_deprecated(c),
                c.get("is_target_subclass", False),
                c["type"] == "Class",
                "Gate" in c.get("base_classes", []),
                len(c["docstring"]),
            ),
        )
        final_concepts.append(best_concept)

        for concept in cluster:
            processed_indices.add(concepts_data.index(concept))
            if concept["name"] != best_concept["name"]:
                logging.info(
                    f"  -> Discarding '{concept['name']}' in favor of '{best_concept['name']}' due to similarity."
                )

    # Add any concepts that were not part of any cluster (singletons)
    for i, concept in enumerate(concepts_data):
        if i not in processed_indices:
            final_concepts.append(concept)

    logging.info(
        f"Reduced concepts from {len(concepts_data)} to {len(final_concepts)}."
    )
    return final_concepts

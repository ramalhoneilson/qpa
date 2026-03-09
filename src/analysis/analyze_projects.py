
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

from src.conf import config

# Define the high-level categories (Taxonomy aligned with user preference)
CATEGORIES = {
    "Full-Stack SDK": "General purpose software development kit (SDK) for creating, compiling, and executing quantum circuits. Includes expansive frameworks like Qiskit, Cirq, PennyLane.",
    "Quantum Machine Learning": "Libraries and tools focused on Quantum Neural Networks (QNNs), parameterized quantum circuits (PQC), and variational quantum classifiers. Integration with ML libraries like PyTorch/TensorFlow.",
    "Algorithms & Libraries": "Collections of specific quantum algorithms (e.g., Shor, Grover) or higher-level libraries built on top of SDKs for specific tasks (not full stacks themselves).",
    "Simulation & Information Theory": "High-performance simulators (statevector, tensor network), quantum information theory tools (entanglement, tomography), and lower-level physics simulations.",
    "Quantum Chemistry & Science": "Domain-specific applications for electronic structure problems, molecular dynamics, and materials science simulation.",
    "Compilers & Languages": "Quantum programming languages (e.g., QASM, Silq), compilers, intermediate representations (IR), and transpilers.",
    "Cloud & Orchestration": "Tools for managing quantum jobs on the cloud, orchestrating hybrids workflows, and accessing remote QPUs.",
    "Hardware Design & Control": "Pulse-level control, qubit calibration, quantum error characterization, chip design (EDA), and interacting directly with hardware control electronics.",
    "Optimization & Finance": "Applications focused on QAOA, quantum annealing, portfolio optimization, and other combinatorial optimization problems.",
    "Error Correction & Mitigation": "Tools for Quantum Error Correction (QEC) codes, decoders, error mitigation techniques (ZNE, CDR), and fault-tolerance analysis."
}

def load_latest_json(data_dir: Path) -> Path:
    """Finds the latest quantum_frameworks_structured_*.json file."""
    files = list(data_dir.glob("quantum_frameworks_structured_*.json"))
    if not files:
        raise FileNotFoundError(f"No structured JSON files found in {data_dir}")
    # Sort by name (timestamp is in the name)
    return sorted(files)[-1]

def analyze_projects():
    print("--- Starting Project Analysis ---")
    
    # 1. Locate Data
    try:
        json_file = load_latest_json(config.RESULTS_DIR)
        print(f"Loading metadata from: {json_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        projects = json.load(f)

    print(f"Loaded {len(projects)} projects.")

    # 2. Load Model
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    # 3. Embed Categories
    category_names = list(CATEGORIES.keys())
    category_descriptions = list(CATEGORIES.values())
    
    # We embed the description of the category to match against the project description
    print("Embedding taxonomy categories...")
    category_embeddings = model.encode(category_descriptions, convert_to_tensor=True)

    # 4. Analyze Projects
    print("Analyzing and categorizing projects...")
    
    analyzed_projects = []
    
    # Prepare batch embedding for efficiency
    project_descriptions = [p.get("description", "") or "" for p in projects]
    project_embeddings = model.encode(project_descriptions, convert_to_tensor=True)

    # Calculate similarities (Result is [num_projects, num_categories])
    # 1 - cosine_distance = cosine_similarity
    similarities = 1 - cdist(project_embeddings.cpu(), category_embeddings.cpu(), 'cosine')

    for i, project in enumerate(projects):
        scores = similarities[i]
        best_idx = scores.argmax()
        best_score = scores[best_idx]
        best_category = category_names[best_idx]
        
        # Heuristic: If confidence is very low, maybe check if name contains strong keywords? 
        # But for now, we trust the embedding with the best score.
        
        analyzed_projects.append({
            **project,
            "predicted_category": best_category,
            "confidence_score": f"{best_score:.4f}"
        })

    # 5. Export to CSV
    output_csv = config.RESULTS_DIR / "quantum_projects_categorized.csv"
    print(f"Exporting results to {output_csv}...")
    
    # Define CSV columns
    fieldnames = [
        "rank", "full_name", "predicted_category", "confidence_score", 
        "stargazers_count", "contributors_count", "last_push", "description", "html_url"
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for p in analyzed_projects:
            # Map JSON keys to CSV columns where necessary or keep as is
            row = {
                "rank": p.get("rank"),
                "full_name": p.get("full_name"),
                "predicted_category": p.get("predicted_category"),
                "confidence_score": p.get("confidence_score"),
                "stargazers_count": p.get("stargazers_count"),
                "contributors_count": p.get("contributors_count"),
                "last_push": p.get("pushed_at", "").split("T")[0], # simplified date
                "description": p.get("description", "").replace("\n", " "),
                "html_url": p.get("html_url")
            }
            writer.writerow(row)

    # 6. Generate Summary Report
    report_file = config.RESULTS_DIR / "report" / "project_analysis_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating report at {report_file}...")
    
    # Count distribution
    distribution = {}
    for p in analyzed_projects:
        cat = p["predicted_category"]
        distribution[cat] = distribution.get(cat, 0) + 1

    # Sort distribution by count
    sorted_dist = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Quantum Projects Analysis Report\n")
        f.write("================================\n\n")
        
        f.write("Category Distribution:\n")
        f.write("----------------------\n")
        for cat, count in sorted_dist:
            f.write(f"{cat}: {count} projects\n")
        
        f.write("\n\nDetailed Classification Confidence (Top 10 High Confidence):\n")
        f.write("------------------------------------------------------------\n")
        
        # Sort by confidence
        sorted_by_conf = sorted(analyzed_projects, key=lambda x: float(x["confidence_score"]), reverse=True)
        
        for p in sorted_by_conf[:10]:
            f.write(f"[{p['confidence_score']}] {p['full_name']} -> {p['predicted_category']}\n")
            
        f.write("\n\nLow Confidence Classifications (Bottom 5 - Potential Misclassifications):\n")
        f.write("-------------------------------------------------------------------------\n")
        for p in sorted_by_conf[-5:]:
            f.write(f"[{p['confidence_score']}] {p['full_name']} -> {p['predicted_category']}\n")

    print("\n--- Analysis Complete ---")
    print(f"Categorized CSV: {output_csv}")
    print(f"Summary Report:  {report_file}")

if __name__ == "__main__":
    analyze_projects()

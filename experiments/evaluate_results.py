import csv
import json
from pathlib import Path
from collections import defaultdict

def normalize_path(path_str):
    # Strip common prefixes to make comparison easier
    path = Path(path_str)
    parts = list(path.parts)
    if parts and (parts[0] == 'qiskit_algorithms' or parts[0] == 'target_github_projects'):
        if parts[0] == 'target_github_projects' and len(parts) > 2:
             # Handle target_github_projects/qiskit-algorithms/qiskit_algorithms/...
             return str(Path(*parts[3:]))
        return str(Path(*parts[1:]))
    return str(path)

def load_manual_labels(file_path):
    labels = set()
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized_file = normalize_path(row['File Name'])
            pattern = row['Pattern Match'].strip()
            labels.add((normalized_file, pattern))
    return labels

def load_tool_predictions(file_path):
    predictions = set()
    with open(file_path, newline='', encoding='utf-8') as f:
        # Tool output uses semicolon delimiter
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            normalized_file = normalize_path(row['file_path'])
            pattern = row['pattern'].strip()
            if pattern != "N/A":
                predictions.add((normalized_file, pattern))
    return predictions

def calculate_metrics(gold_set, pred_set):
    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def per_pattern_metrics(gold_set, pred_set):
    patterns = set([p[1] for p in gold_set] + [p[1] for p in pred_set])
    results = {}
    for pattern in sorted(patterns):
        p_gold = {item for item in gold_set if item[1] == pattern}
        p_pred = {item for item in pred_set if item[1] == pattern}
        results[pattern] = calculate_metrics(p_gold, p_pred)
    return results

def main():
    base_dir = Path(__file__).resolve().parent.parent
    manual_csv = base_dir / "pattern_matches_manual.csv"
    tool_csv = base_dir / "experiments" / "results" / "tool_predictions_qiskit_algorithms.csv"
    results_dir = base_dir / "experiments" / "results"
    
    print(f"Loading manual labels from {manual_csv}")
    gold_set = load_manual_labels(manual_csv)
    
    print(f"Loading tool predictions from {tool_csv}")
    pred_set = load_tool_predictions(tool_csv)
    
    overall = calculate_metrics(gold_set, pred_set)
    by_pattern = per_pattern_metrics(gold_set, pred_set)
    
    # Save results
    with open(results_dir / "evaluation_metrics.json", "w") as f:
        json.dump({"overall": overall, "per_pattern": by_pattern}, f, indent=4)
        
    print("\n--- Overall Metrics ---")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1 Score:  {overall['f1_score']:.4f}")
    
    print("\n--- Top Patterns by F1 ---")
    sorted_patterns = sorted(by_pattern.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    for pattern, metrics in sorted_patterns[:10]:
        if metrics['f1_score'] > 0:
            print(f"{pattern:30} | F1: {metrics['f1_score']:.4f} | TP: {metrics['true_positives']}")

    # Generate Markdown Report
    report_path = results_dir / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# QPA Tool Evaluation Report\n\n")
        f.write("## Summary Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| True Positives | {overall['true_positives']} |\n")
        f.write(f"| False Positives | {overall['false_positives']} |\n")
        f.write(f"| False Negatives | {overall['false_negatives']} |\n")
        f.write(f"| Precision | {overall['precision']:.4f} |\n")
        f.write(f"| Recall | {overall['recall']:.4f} |\n")
        f.write(f"| F1 Score | {overall['f1_score']:.4f} |\n\n")
        
        f.write("## Per-Pattern Metrics\n\n")
        f.write("| Pattern | Precision | Recall | F1 Score | TP | FP | FN |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for pattern, m in sorted(by_pattern.items()):
            f.write(f"| {pattern} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['true_positives']} | {m['false_positives']} | {m['false_negatives']} |\n")

    print(f"\nEvaluation complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()

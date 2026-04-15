"""
Evaluation pipeline for qiskit-finance.

Run:
    just evaluate-qiskit-finance
or:
    python -m src.evaluation.pipelines.evaluate_qiskit_finance
"""

from src.evaluation.evaluate import evaluate

PROJECT      = "qiskit-finance"
TARGET_DIR   = "target_github_projects/qiskit-finance"
GT_PATH      = "data/qiskit_finance_ground_truth.csv"
OUTPUT_PATH  = "data/qiskit_finance_eval_output.csv"
REPORT_PATH  = "docs/qiskit_finance_precision_recall.md"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    evaluate(
        project=PROJECT,
        target_dir=TARGET_DIR,
        gt_path=GT_PATH,
        output_path=OUTPUT_PATH,
        report_path=REPORT_PATH,
        skip_analysis=args.skip_analysis,
    )

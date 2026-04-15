"""
Evaluation pipeline for Qrisp.

Run:
    just evaluate-qrisp
or:
    python -m src.evaluation.pipelines.evaluate_qrisp
"""

from src.evaluation.evaluate import evaluate

PROJECT      = "Qrisp"
TARGET_DIR   = "target_github_projects/Qrisp"
GT_PATH      = "data/qrisp_ground_truth.csv"
OUTPUT_PATH  = "data/qrisp_eval_output.csv"
REPORT_PATH  = "docs/qrisp_precision_recall.md"

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

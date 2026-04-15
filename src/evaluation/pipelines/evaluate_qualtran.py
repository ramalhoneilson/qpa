"""
Evaluation pipeline for Qualtran.

Run:
    just evaluate-qualtran
or:
    python -m src.evaluation.pipelines.evaluate_qualtran
"""

from src.evaluation.evaluate import evaluate

PROJECT      = "Qualtran"
TARGET_DIR   = "target_github_projects/Qualtran"
GT_PATH      = "data/qualtran_ground_truth.csv"
OUTPUT_PATH  = "data/qualtran_run_analysis_output.csv"
REPORT_PATH  = "docs/qualtran_precision_recall.md"

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

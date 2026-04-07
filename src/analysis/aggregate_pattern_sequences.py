import csv
import argparse
from collections import defaultdict
from pathlib import Path

from src.conf import config

INPUT_CSV = config.RESULTS_DIR / "quantum_concept_matches_with_patterns.csv"
OUTPUT_CSV = config.RESULTS_DIR / "file_pattern_sequences.csv"

def main(input_file: Path, output_file: Path):
    if not input_file.exists():
        print(f"Input file {input_file} not found.")
        return

    # file_path -> list of patterns
    file_patterns = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            file_path = row.get("file_path")
            pattern = row.get("pattern")
            
            if file_path and pattern and pattern != "N/A":
                # Keep distinct patterns in the order they were processed for each file
                if pattern not in file_patterns[file_path]:
                    file_patterns[file_path].append(pattern)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["file_path", "pattern_sequence"])
        
        for file_path, patterns in file_patterns.items():
            sequence = " -> ".join(patterns)
            writer.writerow([file_path, sequence])
            
    print(f"Aggregated pattern sequences for {len(file_patterns)} files.")
    print(f"Results saved to {output_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate patterns into a sequence per file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_CSV),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_CSV),
        help="Path for the output CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(Path(args.input).resolve(), Path(args.output).resolve())

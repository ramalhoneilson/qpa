import pandas as pd
from src.conf import config

KB_DIR = config.RESULTS_DIR / "knowledge_base"


# Define the input files and the framework name to be assigned to each
INPUT_FILES = {
    "classiq": KB_DIR / "enriched_classiq_quantum_patterns.csv",
    "pennylane": KB_DIR / "enriched_pennylane_quantum_patterns.csv",
    "qiskit": KB_DIR / "enriched_qiskit_quantum_patterns.csv",
}

# Define the path for the final consolidated file
OUTPUT_FILE = KB_DIR / "knowledge_base.csv"


def consolidate_knowledge_base():
    """
    Consolidates multiple framework-specific CSV files into a single knowledge base file,
    adding a 'framework' column to identify the source of each entry.
    """
    all_dataframes = []

    print("Starting consolidation process...")

    # Loop through each file, read it, and add the framework column
    for framework, file_path in INPUT_FILES.items():
        if not file_path.exists():
            print(f"Warning: Input file not found, skipping: {file_path}")
            continue

        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Add the new 'framework' column with the corresponding name
            df["framework"] = framework

            all_dataframes.append(df)
            print(f"  - Successfully processed {file_path} ({len(df)} rows)")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Check if any dataframes were loaded
    if not all_dataframes:
        print("No dataframes were loaded. Halting process.")
        return

    # Concatenate all the dataframes into a single one
    consolidated_df = pd.concat(all_dataframes, ignore_index=True)

    # Optional: Move the 'framework' column to be the first column for clarity
    cols = list(consolidated_df.columns)
    # This clever line moves 'framework' to the front
    cols.insert(0, cols.pop(cols.index("framework")))
    consolidated_df = consolidated_df[cols]

    # Save the final consolidated dataframe to the output CSV file
    try:
        # Ensure the parent directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        consolidated_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\nConsolidation complete!")
        print(f"Total rows in consolidated file: {len(consolidated_df)}")
        print(f"Output saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"\nError writing to output file {OUTPUT_FILE}: {e}")


if __name__ == "__main__":
    consolidate_knowledge_base()

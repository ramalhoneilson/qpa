"""
Consolidates multiple framework-specific CSV files into a single knowledge base.

This module consolidates enriched pattern data from different quantum computing
frameworks into a unified knowledge base for analysis.
"""

import pandas as pd
from typing import Dict

from src.conf import config


class KnowledgeBaseConsolidator:
    """Consolidates framework-specific pattern data into a unified knowledge base."""
    
    def __init__(self):
        """Initialize the consolidator with file paths."""
        self.kb_dir = config.RESULTS_DIR / "knowledge_base"
        
        # Define the input files and the framework name to be assigned to each
        self.input_files = {
            "classiq": self.kb_dir / "enriched_classiq_quantum_patterns.csv",
            "pennylane": self.kb_dir / "enriched_pennylane_quantum_patterns.csv",
            "qiskit": self.kb_dir / "enriched_qiskit_quantum_patterns.csv",
        }
        
        # Define the path for the final consolidated file
        self.output_file = self.kb_dir / "knowledge_base.csv"

    def consolidate_knowledge_base(self):
        """
        Consolidates multiple framework-specific CSV files into a single knowledge base file,
        adding a 'framework' column to identify the source of each entry.
        """
        all_dataframes = []

        print("Starting knowledge base consolidation process...")

        # Loop through each file, read it, and add the framework column
        for framework, file_path in self.input_files.items():
            if not file_path.exists():
                print(f"Warning: Input file not found, skipping: {file_path}")
                continue

            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Add framework column
                df['framework'] = framework
                
                # Add to our list
                all_dataframes.append(df)
                
                print(f"Successfully processed {framework}: {len(df)} entries")
                
            except Exception as e:
                print(f"Error processing {framework} file: {e}")
                continue

        if not all_dataframes:
            print("No dataframes to consolidate. Exiting.")
            return

        # Concatenate all dataframes
        try:
            consolidated_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Ensure output directory exists
            self.kb_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the consolidated dataframe
            consolidated_df.to_csv(self.output_file, index=False)
            
            print(f"Consolidation complete!")
            print(f"Total entries: {len(consolidated_df)}")
            print(f"Output file: {self.output_file}")
            
            # Print summary by framework
            print("\nSummary by framework:")
            framework_counts = consolidated_df['framework'].value_counts()
            for framework, count in framework_counts.items():
                print(f"  {framework}: {count} entries")
                
        except Exception as e:
            print(f"Error during consolidation: {e}")

    def get_consolidated_data(self) -> pd.DataFrame:
        """Load and return the consolidated knowledge base data."""
        if not self.output_file.exists():
            print(f"Consolidated knowledge base not found: {self.output_file}")
            return pd.DataFrame()
        
        try:
            return pd.read_csv(self.output_file)
        except Exception as e:
            print(f"Error loading consolidated knowledge base: {e}")
            return pd.DataFrame()

    def get_framework_data(self, framework: str) -> pd.DataFrame:
        """Get data for a specific framework from the consolidated knowledge base."""
        df = self.get_consolidated_data()
        if df.empty:
            return df
        
        return df[df['framework'] == framework] if 'framework' in df.columns else pd.DataFrame()

    def get_pattern_counts(self) -> Dict[str, int]:
        """Get pattern counts by framework."""
        df = self.get_consolidated_data()
        if df.empty or 'framework' not in df.columns:
            return {}
        
        return df['framework'].value_counts().to_dict()


def main():
    """Main function to consolidate the knowledge base."""
    consolidator = KnowledgeBaseConsolidator()
    consolidator.consolidate_knowledge_base()


def consolidate_knowledge_base():
    """Convenience function for external calls."""
    main()


if __name__ == "__main__":
    main()
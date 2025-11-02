"""
QPA: Quantum Patterns Analyser Workflow using Prefect

This module defines the complete workflow for quantum pattern analysis
using Prefect for orchestration, dependency management, and monitoring.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@task(name="download-patterns", retries=2, retry_delay_seconds=30)
def download_patterns() -> Dict[str, Any]:
    """Download quantum patterns from PlanQK Atlas."""
    logger = get_run_logger()
    logger.info("Starting pattern download...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.data_acquisition.download_patterns"
        ], capture_output=True, text=True, check=True)
        
        logger.info("Pattern download completed successfully")
        return {
            "status": "success",
            "output": result.stdout,
            "patterns_file": "data/quantum_patterns.json"
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Pattern download failed: {e.stderr}")
        raise


@task(name="discover-projects", retries=2, retry_delay_seconds=30)
def discover_projects() -> Dict[str, Any]:
    """Discover quantum projects from GitHub."""
    logger = get_run_logger()
    logger.info("Starting project discovery...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.data_acquisition.discover_projects"
        ], capture_output=True, text=True, check=True)
        
        logger.info("Project discovery completed successfully")
        return {
            "status": "success",
            "output": result.stdout,
            "projects_file": "data/filtered_repo_list.txt"
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Project discovery failed: {e.stderr}")
        raise


@task(name="extract-notebooks", retries=2, retry_delay_seconds=30)
def extract_notebooks(discovery_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract notebooks from discovered projects."""
    logger = get_run_logger()
    logger.info("Starting notebook extraction...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.preprocessing.extract_notebooks"
        ], capture_output=True, text=True, check=True)
        
        logger.info("Notebook extraction completed successfully")
        return {
            "status": "success",
            "output": result.stdout,
            "notebooks_dir": "notebooks/"
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Notebook extraction failed: {e.stderr}")
        raise


@task(name="convert-notebooks", retries=2, retry_delay_seconds=30)
def convert_notebooks(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert notebooks from .ipynb to .py files."""
    logger = get_run_logger()
    logger.info("Starting notebook conversion...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.preprocessing.convert_notebooks"
        ], capture_output=True, text=True, check=True)
        
        logger.info("Notebook conversion completed successfully")
        return {
            "status": "success",
            "output": result.stdout,
            "converted_dir": "converted_notebooks/"
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Notebook conversion failed: {e.stderr}")
        raise


@task(name="extract-concepts", retries=2, retry_delay_seconds=30)
def extract_concepts(discovery_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract quantum concepts from frameworks."""
    logger = get_run_logger()
    logger.info("Starting concept extraction...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.extraction.extract_concepts"
        ], capture_output=True, text=True, check=True)
        
        logger.info("Concept extraction completed successfully")
        return {
            "status": "success",
            "output": result.stdout,
            "concept_files": [
                "data/classiq_quantum_concepts.csv",
                "data/pennylane_quantum_concepts.csv",
                "data/qiskit_quantum_concepts.csv"
            ]
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Concept extraction failed: {e.stderr}")
        raise


@task(name="run-analysis", retries=2, retry_delay_seconds=60)
def run_analysis(
    patterns_result: Dict[str, Any],
    conversion_result: Dict[str, Any],
    concepts_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the main semantic analysis."""
    logger = get_run_logger()
    logger.info("Starting main analysis...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.analysis.run_analysis"
        ], capture_output=True, text=True, check=True)
        
        logger.info("Main analysis completed successfully")
        return {
            "status": "success",
            "output": result.stdout,
            "analysis_file": "data/quantum_concept_matches_with_patterns.csv"
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Main analysis failed: {e.stderr}")
        raise


@task(name="generate-report", retries=2, retry_delay_seconds=30)
def generate_report(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the final analysis report."""
    logger = get_run_logger()
    logger.info("Starting report generation...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.analysis.generate_report"
        ], capture_output=True, text=True, check=True)
        
        logger.info("Report generation completed successfully")
        return {
            "status": "success",
            "output": result.stdout,
            "report_files": [
                "data/final_pattern_report.txt",
                "docs/final_pattern_report.md",
                "data/report/"
            ]
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Report generation failed: {e.stderr}")
        raise


@task(name="generate-experimental-data", retries=2, retry_delay_seconds=30)
def generate_experimental_data(report_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate experimental data report."""
    logger = get_run_logger()
    logger.info("Starting experimental data generation...")
    
    try:
        result = subprocess.run([
            sys.executable, "-c", "from src.reporting import generate_experimental_data_report; generate_experimental_data_report()"
        ], capture_output=True, text=True, check=True)
        
        logger.info("Experimental data generation completed successfully")
        return {
            "status": "success",
            "output": result.stdout,
            "experimental_file": "docs/experimental_data.md"
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Experimental data generation failed: {e.stderr}")
        raise


@flow(
    name="qpa-analysis",
    description="Complete Quantum Patterns Analyser (QPA) workflow",
    retries=1,
    retry_delay_seconds=60
)
def qpa_flow() -> Dict[str, Any]:
    """
    Main workflow for QPA: Quantum Patterns Analyser.
    
    This flow orchestrates the complete analysis pipeline:
    1. Download patterns from PlanQK Atlas
    2. Discover quantum projects from GitHub
    3. Extract notebooks from projects
    4. Convert notebooks to Python files
    5. Extract concepts from frameworks
    6. Run main semantic analysis
    7. Generate final report
    8. Generate experimental data report
    """
    logger = get_run_logger()
    logger.info("Starting QPA: Quantum Patterns Analyser Workflow")
    
    # Step 1-2: Data Acquisition (parallel)
    patterns_result = download_patterns()
    projects_result = discover_projects()
    
    # Step 3-4: Preprocessing (sequential)
    notebooks_result = extract_notebooks(projects_result)
    conversion_result = convert_notebooks(notebooks_result)
    
    # Step 5: Extraction (depends on projects discovery)
    concepts_result = extract_concepts(projects_result)
    
    # Step 7-8: Analysis (depends on patterns, conversion, and concepts)
    analysis_result = run_analysis(patterns_result, conversion_result, concepts_result)
    report_result = generate_report(analysis_result)
    
    # Generate experimental data
    experimental_result = generate_experimental_data(report_result)
    
    # Create workflow summary
    summary = f"""
# QPA: Quantum Patterns Analyser Workflow Summary

## Workflow Status: COMPLETED

### Results:
- **Patterns Downloaded**: {patterns_result['patterns_file']}
- **Projects Discovered**: {projects_result['projects_file']}
- **Notebooks Extracted**: {notebooks_result['notebooks_dir']}
- **Notebooks Converted**: {conversion_result['converted_dir']}
- **Concepts Extracted**: {len(concepts_result['concept_files'])} files
- **Analysis Completed**: {analysis_result['analysis_file']}
- **Report Generated**: {len(report_result['report_files'])} files
- **Experimental Data**: {experimental_result['experimental_file']}

### Next Steps:
1. Review the generated reports in `docs/final_pattern_report.md`
2. Check experimental data in `docs/experimental_data.md`
3. Analyze results in `data/report/` directory

Workflow completed successfully!
    """
    
    create_markdown_artifact(
        key="workflow-summary",
        markdown=summary,
        description="QPA: Quantum Patterns Analyser Workflow Summary"
    )
    
    logger.info("QPA: Quantum Patterns Analyser Workflow completed successfully!")
    
    return {
        "status": "success",
        "patterns": patterns_result,
        "projects": projects_result,
        "notebooks": notebooks_result,
        "conversion": conversion_result,
        "concepts": concepts_result,
        "analysis": analysis_result,
        "report": report_result,
        "experimental": experimental_result
    }


if __name__ == "__main__":
    # Run the workflow
    result = qpa_flow()
    print(f"Workflow completed with status: {result['status']}")

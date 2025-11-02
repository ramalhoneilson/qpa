# Reporting module for QPA: Quantum Patterns Analyser

"""
This module contains all reporting functionality for the QPA: Quantum Patterns Analyser project.

The reporting module provides:
- Unified report generation system
- Pattern analysis utilities
- PDF generation capabilities
- Comprehensive data export

Key Classes:
- ReportGenerator: Unified system for all report types
- PatternAnalyzer: Pattern coverage and statistics analysis
- PDFGenerator: Markdown to PDF conversion utilities
"""

from .report_generator import (
    ReportGenerator,
    generate_experimental_data_report,
    generate_base_concept_report,
    generate_pattern_report,
    generate_extended_pattern_analysis,
    generate_all_reports,
    main as generate_reports_main,
)

from .pattern_analyzer import (
    PatternAnalyzer,
    analyze_pattern_coverage,
    get_pattern_statistics,
    export_pattern_analysis,
    main as analyze_patterns_main,
)

from .pdf_generator import (
    PDFGenerator,
    generate_pdfs,
    main as generate_pdfs_main,
)

__all__ = [
    # Report Generator
    "ReportGenerator",
    "generate_experimental_data_report",
    "generate_base_concept_report", 
    "generate_pattern_report",
    "generate_extended_pattern_analysis",
    "generate_all_reports",
    "generate_reports_main",
    
    # Pattern Analyzer
    "PatternAnalyzer",
    "analyze_pattern_coverage",
    "get_pattern_statistics",
    "export_pattern_analysis",
    "analyze_patterns_main",
    
    # PDF Generator
    "PDFGenerator",
    "generate_pdfs",
    "generate_pdfs_main",
]
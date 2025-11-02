#!/usr/bin/env python3
"""
Test runner script for the QPA (Quantum Patterns Analyser) project.

This script provides a convenient way to run tests with different configurations
and options.
"""

import argparse
import subprocess
import sys


def run_tests(
    test_path=None, verbose=False, coverage=False, markers=None, parallel=False
):
    """
    Run tests with specified options.

    Args:
        test_path: Specific test file or directory to run
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        markers: Test markers to filter by
        parallel: Run tests in parallel
    """
    cmd = ["python", "-m", "pytest"]

    if test_path:
        cmd.append(str(test_path))
    else:
        cmd.append("tests/")

    if verbose:
        cmd.extend(["-v", "-s"])

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    if markers:
        cmd.extend(["-m", markers])

    if parallel:
        cmd.extend(["-n", "auto"])

    # Add common options
    cmd.extend(["--tb=short", "--strict-markers", "--color=yes"])

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with return code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest.")
        return 1


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Run tests for QPA (Quantum Patterns Analyser) project"
    )
    parser.add_argument(
        "test_path", nargs="?", help="Specific test file or directory to run"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-c", "--coverage", action="store_true", help="Enable coverage reporting"
    )
    parser.add_argument(
        "-m",
        "--markers",
        help="Test markers to filter by (e.g., 'unit', 'integration')",
    )
    parser.add_argument(
        "-p", "--parallel", action="store_true", help="Run tests in parallel"
    )
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--core-concepts", action="store_true", help="Run only core concepts tests"
    )

    args = parser.parse_args()

    # Determine test path
    test_path = args.test_path
    if not test_path:
        if args.unit:
            test_path = "tests/test_identify_classiq_core_concepts.py::TestCreateSummaryFromDocstring"
        elif args.integration:
            test_path = "tests/test_identify_classiq_core_concepts.py::TestIntegration"
        elif args.core_concepts:
            test_path = "tests/test_identify_classiq_core_concepts.py"

    # Determine markers
    markers = args.markers
    if args.unit and not markers:
        markers = "unit"
    elif args.integration and not markers:
        markers = "integration"
    elif args.core_concepts and not markers:
        markers = "core_concepts"

    # Run tests
    return run_tests(
        test_path=test_path,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=markers,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    sys.exit(main())

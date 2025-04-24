#!/usr/bin/env python3
"""
Test runner for baseline tests.
This script runs all baseline tests and generates a report.
"""
import os
import sys
import unittest
import pytest
import time
import argparse
from pathlib import Path
from datetime import datetime
import json


def run_unittest_tests(test_pattern, verbose=False):
    """Run unittest-based tests."""
    loader = unittest.TestLoader()
    tests = loader.discover(".", pattern=test_pattern)

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(tests)

    return result


def run_pytest_tests(test_pattern, verbose=False):
    """Run pytest-based tests."""
    # Find test files matching the pattern
    tests = list(Path(".").glob(test_pattern))
    test_paths = [str(test) for test in tests]

    # If no test files found, return success (we'll rely on unittest)
    if not test_paths:
        print("No pytest tests found, skipping pytest runner")
        return 0

    # Create args for pytest
    args = ["--asyncio-mode=auto"]
    if verbose:
        args.append("-v")
    args.extend(test_paths)

    return pytest.main(args)


def create_report(unittest_result, pytest_result, output_dir):
    """Create a test report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"baseline_test_report_{timestamp}.json"

    # Create report data
    report = {
        "timestamp": datetime.now().isoformat(),
        "unittest_results": {
            "run": unittest_result.testsRun,
            "failures": len(unittest_result.failures),
            "errors": len(unittest_result.errors),
            "skipped": len(unittest_result.skipped),
        },
        "pytest_results": {
            "status": pytest_result,  # exit code from pytest.main()
            "message": get_pytest_status_message(pytest_result),
        },
        "overall_status": (
            "PASS" if unittest_result.wasSuccessful() and pytest_result == 0 else "FAIL"
        ),
    }

    # Add detailed failure information
    if not unittest_result.wasSuccessful():
        report["unittest_details"] = {
            "failures": [
                {"test": str(test), "message": err}
                for test, err in unittest_result.failures
            ],
            "errors": [
                {"test": str(test), "message": err}
                for test, err in unittest_result.errors
            ],
        }

    # Write the report
    os.makedirs(Path(output_dir), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_path


def get_pytest_status_message(code):
    """Convert pytest exit code to message."""
    messages = {
        0: "All tests passed",
        1: "Some tests failed",
        2: "Test execution was interrupted",
        3: "Internal pytest error",
        4: "pytest command line usage error",
        5: "No tests were collected",
    }
    return messages.get(code, f"Unknown status code: {code}")


def print_report_summary(report_path):
    """Print a summary of the test report."""
    with open(report_path, "r") as f:
        report = json.load(f)

    print("\n" + "=" * 50)
    print(f"BASELINE TEST SUMMARY - {report['timestamp']}")
    print("=" * 50)
    print(f"Overall Status: {report['overall_status']}")
    print("\nUnittest Results:")
    print(f"  Tests Run: {report['unittest_results']['run']}")
    print(f"  Failures: {report['unittest_results']['failures']}")
    print(f"  Errors: {report['unittest_results']['errors']}")
    print(f"  Skipped: {report['unittest_results']['skipped']}")
    print("\nPytest Results:")
    print(f"  Status: {report['pytest_results']['message']}")

    if report["overall_status"] == "FAIL":
        print("\nFailure Details:")
        if "unittest_details" in report:
            if report["unittest_details"].get("failures"):
                print("\nFailures:")
                for i, failure in enumerate(report["unittest_details"]["failures"], 1):
                    print(f"  {i}. {failure['test']}")

            if report["unittest_details"].get("errors"):
                print("\nErrors:")
                for i, error in enumerate(report["unittest_details"]["errors"], 1):
                    print(f"  {i}. {error['test']}")

    print("\nFull report saved to:", report_path)
    print("=" * 50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run baseline tests for GNN package")
    parser.add_argument(
        "--output",
        "-o",
        default="reports",
        help="Directory for test reports (default: 'reports')",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="test_*_baseline.py",
        help="Test file pattern (default: 'test_*_baseline.py')",
    )
    parser.add_argument(
        "--unittest-only",
        action="store_true",
        help="Run only unittest tests (skip pytest)",
    )
    parser.add_argument(
        "--pytest-only",
        action="store_true",
        help="Run only pytest tests (skip unittest)",
    )
    return parser.parse_args()


def main():
    """Main function to run all tests."""
    args = parse_args()

    print(f"Running baseline tests with pattern: {args.pattern}")
    start_time = time.time()

    # Run unittest tests if not pytest-only
    if not args.pytest_only:
        print("\nRunning unittest tests...")
        unittest_result = run_unittest_tests(args.pattern, args.verbose)
    else:
        # Create a mock successful result
        unittest_result = unittest.TestResult()

    # Run pytest tests if not unittest-only
    if not args.unittest_only:
        print("\nRunning pytest tests...")
        pytest_result = run_pytest_tests(args.pattern, args.verbose)
    else:
        # Skip pytest with a success code
        pytest_result = 0

    # Create report
    report_path = create_report(unittest_result, pytest_result, args.output)

    # Print summary
    print_report_summary(report_path)

    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    # Return exit code
    return 0 if unittest_result.wasSuccessful() and pytest_result == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

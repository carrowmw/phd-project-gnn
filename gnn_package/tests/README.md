# Baseline Testing Suite

This directory contains baseline tests for the GNN package and private_uoapi module. These tests serve as a reference for current functionality before any code cleanup or refactoring to ensure changes don't break existing behavior.

## Test Files

The baseline testing suite includes the following test files:

1. **test_config_baseline.py**: Tests the configuration system functionality
2. **test_data_processing_baseline.py**: Tests data loading and preprocessing
3. **test_model_baseline.py**: Tests model creation, training, and prediction
4. **test_integration_baseline.py**: Tests end-to-end workflows and entry points
5. **test_api_interface_baseline.py**: Tests the private_uoapi wrapper functionality

## Running the Tests

You can run the tests using the provided script:

```bash
python run_baseline_tests.py
```

This will run all baseline tests and generate a report in the `reports` directory.

### Command-line Options

The test runner supports several options:

- `--output` or `-o`: Specify the output directory for test reports (default: 'reports')
- `--verbose` or `-v`: Enable verbose output for more detailed test information
- `--pattern` or `-p`: Specify a custom test file pattern (default: 'test_*_baseline.py')

Example:

```bash
python run_baseline_tests.py --verbose --output ./test_reports
```

## Test Reports

After running the tests, a JSON report will be generated with details about the test execution. The report includes:

- Timestamp of the test run
- Overall test status (PASS/FAIL)
- Summary of unittest results (tests run, failures, errors, skipped)
- Summary of pytest results
- Detailed failure information if any tests failed

Example report structure:

```json
{
  "timestamp": "2025-04-23T15:30:45",
  "unittest_results": {
    "run": 32,
    "failures": 0,
    "errors": 0,
    "skipped": 1
  },
  "pytest_results": {
    "status": 0,
    "message": "All tests passed"
  },
  "overall_status": "PASS"
}
```

## Writing New Baseline Tests

When writing new baseline tests, follow these guidelines:

1. Name the test file with the pattern `test_*_baseline.py` for automatic discovery
2. Use unittest for synchronous tests and pytest for asynchronous tests
3. Document the purpose of each test with docstrings
4. Test the current behavior, not the ideal behavior
5. Create small, focused tests that test a single aspect of functionality
6. Use appropriate setup and teardown methods for test isolation

## Test Dependencies

The tests require the following dependencies:

- unittest (standard library)
- pytest
- pytest-asyncio
- httpx (for API tests)

Install the dependencies with:

```bash
pip install pytest pytest-asyncio httpx
```

## Note on Test Coverage

These baseline tests focus on verifying the functionality of major entry points and components rather than achieving full code coverage. They are designed to catch unintended changes in behavior during refactoring efforts.

After the initial cleanup phase, consider expanding the test suite to improve coverage.
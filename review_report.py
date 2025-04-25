import pandas as pd

# Load and process reports
vulture_results = pd.read_csv(
    "vulture_report.txt", sep=":", names=["file", "line", "message"]
)
coverage_results = pd.read_html("htmlcov/index.html")[0]

# Identify low-coverage modules
low_coverage = coverage_results[coverage_results["Cover"] < 50]

# Create final report
print("=== Modules with low test coverage ===")
print(low_coverage[["Module", "Cover"]].to_string(index=False))

print("\n=== Potentially unused code ===")
print(vulture_results[["file", "line", "message"]].to_string(index=False))

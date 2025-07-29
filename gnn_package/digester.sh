#!/bin/bash

# digester.sh - Script to ingest only Python files from codebase recursively
# This is a simpler implementation that doesn't rely on gitingest's recursive behavior

set -e  # Exit on error

# Configuration
MAX_FILE_SIZE_KB=500  # Set maximum file size to 500 KB
MAX_FILE_SIZE_BYTES=$((MAX_FILE_SIZE_KB * 1024))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="$PROJECT_ROOT/digested_gnn_package.txt"

echo "Starting recursive Python codebase ingestion from gnn_package directory..."
echo "- Max file size: ${MAX_FILE_SIZE_KB}KB"
echo "- Output will be saved to: ${OUTPUT_FILE}"

# Create/clear the output file
> "$OUTPUT_FILE"

# Add header
echo "# GNN Package Python Files" > "$OUTPUT_FILE"
echo "Generated on $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Find all Python files, excluding __pycache__ directories
PYTHON_FILES=$(find "$SCRIPT_DIR" -type f -name "*.py" -not -path "*/__pycache__/*")
TOTAL_FILES=$(echo "$PYTHON_FILES" | wc -l)
echo "Found $TOTAL_FILES Python files to process"

# Process each Python file
for file in $PYTHON_FILES; do
    # Get file size in bytes
    file_size=$(stat -f%z "$file" 2>/dev/null || stat --format="%s" "$file")

    # Skip files larger than max size
    if [ "$file_size" -gt "$MAX_FILE_SIZE_BYTES" ]; then
        echo "Skipping large file: $file ($(($file_size / 1024)) KB)"
        continue
    fi

    # Get relative path for display
    rel_path=${file#$PROJECT_ROOT/}

    # Add file separator and header
    echo -e "\n================================================" >> "$OUTPUT_FILE"
    echo "File: $rel_path" >> "$OUTPUT_FILE"
    echo -e "================================================\n" >> "$OUTPUT_FILE"

    # Append file content
    cat "$file" >> "$OUTPUT_FILE"
done

# Add summary at the end
echo -e "\n\n# Summary" >> "$OUTPUT_FILE"
echo "Total Python files processed: $(grep -c "^File: " "$OUTPUT_FILE")" >> "$OUTPUT_FILE"
echo "Total size: $(du -h "$OUTPUT_FILE" | cut -f1)" >> "$OUTPUT_FILE"

echo "Digestion complete! Output saved to $OUTPUT_FILE"
echo "Processed $(grep -c "^File: " "$OUTPUT_FILE") Python files"
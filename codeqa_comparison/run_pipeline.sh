#!/usr/bin/env bash
# Run the full file-identification pipeline in order.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Step 1/4: Identifying repos ==="
uv run python identify_files.py identify_repos

echo "=== Step 2/4: Cloning repos ==="
uv run python identify_files.py clone_repos

echo "=== Step 3/4: Identifying files ==="
uv run python identify_files.py identify_files

echo "=== Step 4/4: Assembling annotated contexts ==="
uv run python identify_files.py assemble_contexts

echo "=== Pipeline complete ==="

#!/bin/bash
set -e

# Environment variables expected:
# - GITHUB_REPOSITORY: e.g., "ChipFlow/jax-spice"
# - GITHUB_SHA: commit to checkout
# - GITHUB_TOKEN: for private repo access (optional for public repos)

echo "=== GPU Test Runner ==="
echo "Repository: ${GITHUB_REPOSITORY}"
echo "Commit: ${GITHUB_SHA}"

cd /app

# Clone the repository at the specific commit
echo "Cloning repository..."
if [ -n "$GITHUB_TOKEN" ]; then
    git clone --depth 1 --recurse-submodules \
        "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
        --branch main source
    cd source
    git fetch --depth 1 origin "$GITHUB_SHA"
    git checkout "$GITHUB_SHA"
else
    git clone --depth 1 --recurse-submodules \
        "https://github.com/${GITHUB_REPOSITORY}.git" \
        --branch main source
    cd source
    git fetch --depth 1 origin "$GITHUB_SHA"
    git checkout "$GITHUB_SHA"
fi

# Update submodules if needed
git submodule update --init --recursive

echo "Installing workspace packages..."
# Install the workspace in the pre-existing venv
uv sync --locked --extra cuda12

echo "Running tests..."
uv run pytest tests/ -v --tb=short -x

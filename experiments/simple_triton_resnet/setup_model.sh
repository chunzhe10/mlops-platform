#!/usr/bin/env bash
# Setup script: download ResNet50 model using the triton add_dummy_model.sh script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==> Running triton/add_dummy_model.sh to download ResNet50 model..."
cd "${REPO_ROOT}/triton"
bash add_dummy_model.sh resnet50 1

echo "==> Model setup complete. Check triton/models/resnet50/"
ls -R "${REPO_ROOT}/triton/models/resnet50/"

#!/usr/bin/env bash
# Download CIFAR-10 dataset for testing ResNet inference

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"

mkdir -p "${DATA_DIR}"

echo "==> Downloading CIFAR-10 dataset..."

# Download CIFAR-10 Python version
CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_TAR="${DATA_DIR}/cifar-10-python.tar.gz"

if [[ -f "${DATA_DIR}/cifar-10-batches-py/data_batch_1" ]]; then
  echo "CIFAR-10 already exists at ${DATA_DIR}/cifar-10-batches-py/"
  exit 0
fi

if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "${CIFAR_TAR}" "${CIFAR_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${CIFAR_TAR}" "${CIFAR_URL}"
else
  echo "Error: need curl or wget to download dataset." >&2
  exit 1
fi

echo "==> Extracting CIFAR-10..."
tar -xzf "${CIFAR_TAR}" -C "${DATA_DIR}"
rm -f "${CIFAR_TAR}"

echo "==> CIFAR-10 downloaded and extracted to ${DATA_DIR}/cifar-10-batches-py/"
ls -lh "${DATA_DIR}/cifar-10-batches-py/" | head -10

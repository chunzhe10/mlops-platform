#!/usr/bin/env bash

# Simplified script: download and extract resnet50 tarball and prepare a Triton model repo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
MODEL_NAME="${1:-resnet50}"
MODEL_VERSION="${2:-1}"

MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}/${MODEL_VERSION}"
mkdir -p "${MODEL_PATH}"

echo "Preparing model repository: ${MODEL_NAME} -> ${MODEL_PATH}"

# Tarball URL (user-provided). Use https if protocol-relative
TARBALL_URL="https://download.onnxruntime.ai/onnx/models/resnet50.tar.gz"

TMP_TAR="$(mktemp /tmp/resnet50.XXXX.tar.gz)"
TMP_DIR="$(mktemp -d /tmp/resnet50_extract.XXXX)"

download_ok=0
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "$TMP_TAR" "$TARBALL_URL" && download_ok=1 || download_ok=0
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TMP_TAR" "$TARBALL_URL" && download_ok=1 || download_ok=0
else
  echo "Error: need curl or wget to download model tarball." >&2
  exit 1
fi

if [[ "$download_ok" -eq 1 ]]; then
  echo "Downloaded tarball to $TMP_TAR — extracting..."
  tar -xzf "$TMP_TAR" -C "$TMP_DIR"
  # find first .onnx file and copy it
  onnx_file="$(find "$TMP_DIR" -type f -iname '*.onnx' | head -n1 || true)"
  if [[ -n "$onnx_file" ]]; then
    cp "$onnx_file" "${MODEL_PATH}/model.onnx"
    echo "Placed ONNX model at ${MODEL_PATH}/model.onnx"
  else
    echo "No ONNX file found in tarball; leaving placeholder in ${MODEL_PATH}" >&2
    echo "(extracted files:)" && find "$TMP_DIR" -maxdepth 3 -type f -print
    echo "This repository will need a model.onnx placed manually."
    cat > "${MODEL_PATH}/model.txt" <<'EOF'
Tarball extracted, but no .onnx found. Place your model as model.onnx here.
EOF
  fi
  rm -f "$TMP_TAR"
  rm -rf "$TMP_DIR"
else
  echo "Warning: download failed; cleaning up temporary files." >&2
  # Remove any partially downloaded/extracted artifacts
  [[ -f "$TMP_TAR" ]] && rm -f "$TMP_TAR" || true
  [[ -d "$TMP_DIR" ]] && rm -rf "$TMP_DIR" || true
  echo "Warning: model tarball could not be downloaded. No placeholder created; place model.onnx manually when available." >&2
fi

# For this ResNet50 model, the actual input/output names from the ONNX are known
# Input: gpu_0/data_0 (shape [-1, 3, 224, 224] - first dim is dynamic batch)
# Output: gpu_0/resnet_node_output_0 (shape [-1, 1000])
cat > "${MODELS_DIR}/${MODEL_NAME}/config.pbtxt" <<EOF
name: "${MODEL_NAME}"
platform: "onnxruntime_onnx"
max_batch_size: 1
input [ {
  name: "gpu_0/data_0"
  data_type: TYPE_FP32
  dims: [ -1, 3, 224, 224 ]
} ]
output [ {
  name: "gpu_0/resnet_node_output_0"
  data_type: TYPE_FP32
  dims: [ -1, 1000 ]
} ]
instance_group [ { kind: KIND_CPU } ]
EOF

echo "Created ${MODELS_DIR}/${MODEL_NAME}/config.pbtxt"
echo "Model repository layout:"
ls -R "${MODELS_DIR}/${MODEL_NAME}"

echo "Done. If you placed model.onnx, Triton can serve it from ${MODELS_DIR}/${MODEL_NAME}/"

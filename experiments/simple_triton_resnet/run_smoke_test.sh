#!/usr/bin/env bash
# End-to-end smoke test using uv venv
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Ensure uv
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> Ensuring Python 3.10 (this may take a few minutes on first run)..."
uv python install 3.10 || true

# Venv (Python 3.10) - remove old one to avoid interactive prompts
VENV="experiments/simple_triton_resnet/.venv"
rm -rf "$VENV"
echo "==> Creating virtual environment..."
uv venv --python 3.10 "$VENV"
. "$VENV/bin/activate"

echo "==> Installing Python dependencies..."
uv pip install -r experiments/simple_triton_resnet/requirements.txt

echo "==> Stopping existing Triton containers..."
# Triton clean start
docker compose --profile triton down || true

echo "==> Setting up ResNet50 model..."
# Model & server
experiments/simple_triton_resnet/setup_model.sh

echo "==> Starting Triton server..."
docker compose --profile triton up -d --build

echo "==> Waiting for Triton to be ready..."
# Wait for health
for i in {1..60}; do
  if curl -s http://localhost:8000/v2/health/ready | grep -q OK; then echo "Triton ready!"; break; fi
  sleep 1
  if [ "$i" -eq 60 ]; then echo "Triton not ready after 60s"; exit 1; fi
done

echo "==> Downloading CIFAR-10 dataset..."
# Dataset
experiments/simple_triton_resnet/download_data.sh

echo "==> Running inference client..."
# Client
python experiments/simple_triton_resnet/client.py --index 0

echo "==> Smoke test complete!"

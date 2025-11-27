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

# Venv
VENV="experiments/simple_triton_resnet/.venv"
uv venv "$VENV"
. "$VENV/bin/activate"
uv pip install -r experiments/simple_triton_resnet/requirements.txt

# Triton clean start
docker compose --profile triton down || true

# Model & server
experiments/simple_triton_resnet/setup_model.sh

docker compose --profile triton up -d --build

# Wait for health
for i in {1..60}; do
  if curl -s http://localhost:8000/v2/health/ready | grep -q OK; then echo "Triton ready"; break; fi
  sleep 1
  if [ "$i" -eq 60 ]; then echo "Triton not ready after 60s"; exit 1; fi
done

# Dataset
experiments/simple_triton_resnet/download_data.sh

# Client
python experiments/simple_triton_resnet/client.py --index 0

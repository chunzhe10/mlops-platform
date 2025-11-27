Simple Triton ResNet — Smoke Test

This experiment downloads a ResNet50 ONNX model into the Triton model repo, pulls a small CIFAR-10 dataset, and sends an inference request to a running Triton server via HTTP/REST.

Prerequisites
- Triton container running from repo root:
  - `docker compose --profile triton up -d --build`
- Python 3 with `pip install pillow requests numpy`

Setup
```bash
# 1) Download model into triton/models/resnet50/1/model.onnx
experiments/simple_triton_resnet/setup_model.sh

# 2) Download CIFAR-10 dataset into data/
experiments/simple_triton_resnet/download_data.sh
```

Run client
```bash
# Default: http://localhost:8000, model=resnet50, uses data/cifar-10-batches-py
python3 experiments/simple_triton_resnet/client.py --index 0

# Specify a different image index
python3 experiments/simple_triton_resnet/client.py --index 123
```

Notes
- This is a smoke test; CIFAR-10 labels do not correspond to ImageNet classes, so the top-5 indices are for connectivity sanity checking.
- If the Triton server runs inside Docker, ensure ports 8000/8001/8002 are open and the model repository is mounted correctly (compose does this already).

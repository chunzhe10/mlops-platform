# Simple Triton ResNet — Smoke Test

End-to-end smoke test for NVIDIA Triton Inference Server with ResNet50 ONNX model.

## Overview

This experiment:
- Downloads ResNet50 ONNX model (97MB) from onnxruntime.ai
- Configures Triton with `max_batch_size: 0` (fixed batch dimension)
- Downloads CIFAR-10 dataset for test images
- Runs inference via official Triton HTTP client
- Validates server connectivity and model serving

**Note:** CIFAR-10 labels don't match ImageNet classes; top-5 predictions are for connectivity testing only.

## Quick Start

**Option 1: Full automated smoke test**
```bash
# Install dependencies first (once)
pip install -r experiments/simple_triton_resnet/requirements.txt

# Run everything: setup model, download data, start Triton, run inference
python experiments/simple_triton_resnet/client.py --smoke-test
```

**Option 2: Step-by-step**
```bash
# 1. Setup (downloads model and dataset)
python experiments/simple_triton_resnet/client.py --setup

# 2. Start Triton manually
docker compose --profile triton up -d --build

# 3. Run inference
python experiments/simple_triton_resnet/client.py --index 0
```

## Manual Setup

The `client.py` script includes built-in setup functions. You can also run each step manually:

### 1. Setup Model and Data

```bash
# All-in-one setup command
python experiments/simple_triton_resnet/client.py --setup
```

This downloads:
- ResNet50 ONNX model to `triton/models/resnet50/1/model.onnx`
- CIFAR-10 dataset to `data/cifar-10-batches-py/`

### 2. Start Triton Server

```bash
# Using Docker Compose profiles
docker compose --profile triton up -d --build
```

### 3. Run Inference

```bash
# Install dependencies
pip install -r experiments/simple_triton_resnet/requirements.txt

# Run inference on image index 0
python experiments/simple_triton_resnet/client.py --index 0

# Try different images (0-9999)
python experiments/simple_triton_resnet/client.py --index 123
```

## Model Configuration

The ResNet50 ONNX model from onnxruntime.ai has:
- **Input:** `gpu_0/data_0` - shape `[1, 3, 224, 224]` (NCHW, FP32)
- **Output:** `gpu_0/softmax_1` - shape `[1, 1000]` (ImageNet classes)
- **Batching:** Disabled (`max_batch_size: 0`) due to fixed batch dimension in ONNX

The client preprocesses CIFAR-10 images (32×32 RGB) by:
1. Resizing to 224×224 using bilinear interpolation
2. Normalizing with ImageNet mean/std
3. Converting to NCHW float32 format

## Troubleshooting

**Triton not starting:**
```bash
# Check container logs
docker compose --profile triton logs triton

# Verify model files exist
ls -lh triton/models/resnet50/1/model.onnx
cat triton/models/resnet50/config.pbtxt
```

**Health check failing:**
```bash
# Test manually
curl -v http://localhost:8000/v2/health/live
curl -v http://localhost:8000/v2/health/ready

# Check model status
curl http://localhost:8000/v2/models/resnet50
```

**Python dependency issues:**
- Use Python 3.10 for best wheel availability
- `geventhttpclient` (Triton dependency) may need compilation on Python 3.11+

## File Structure

```
experiments/simple_triton_resnet/
├── README.md              # Documentation
├── client.py              # All-in-one: inference + setup + smoke test
└── requirements.txt       # Python dependencies
```

The client supports multiple modes:
- `python client.py --index 0` - Run inference
- `python client.py --setup` - Download model and data
- `python client.py --smoke-test` - Full end-to-end test

## Suggested Improvements

### Organization
1. **Add `.gitignore` for experiment:**
   ```
   .venv/
   *.pyc
   __pycache__/
   .pytest_cache/
   ```

2. **Create `tests/` directory:**
   - Add unit tests for preprocessing functions
   - Add integration test fixture with mock responses
   - Use pytest for test automation

### Code Quality
1. **Extract preprocessing to separate module:**
   - `preprocessing.py` with reusable functions
   - Easier to unit test and share across experiments

2. **Add configuration file:**
   - `config.yaml` with server URL, model name, tensor names
   - Environment-specific configs (dev/staging/prod)

3. **Add logging:**
   - Use Python `logging` instead of `print()`
   - Configurable log levels (DEBUG/INFO/WARNING)

### Additional Enhancements
1. **Add performance metrics:**
   - Measure inference latency
   - Track throughput (requests/sec)
   - Log to CSV for analysis

2. **Support batch inference:**
   - Send multiple images in one request
   - Compare single vs batch performance

3. **CI/CD integration:**
   - GitHub Actions workflow to run smoke test
   - Fail PR if Triton inference breaks

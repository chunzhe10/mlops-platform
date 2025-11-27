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

Run the automated smoke test:

```bash
# From repository root
bash experiments/simple_triton_resnet/run_smoke_test.sh
```

This script:
1. Creates Python 3.10 virtual environment with `uv`
2. Installs `tritonclient[http]` and dependencies
3. Downloads ResNet50 model to `triton/models/resnet50/`
4. Builds and starts Triton container
5. Downloads CIFAR-10 dataset to `data/`
6. Runs inference on test image
7. Prints top-5 class predictions

## Manual Setup

### 1. Setup Model

```bash
# Downloads ResNet50 ONNX to triton/models/resnet50/1/model.onnx
# Generates config.pbtxt with correct tensor names and shapes
bash experiments/simple_triton_resnet/setup_model.sh
```

### 2. Start Triton Server

```bash
# Using Docker Compose profiles
docker compose --profile triton up -d --build
```

### 3. Download Test Dataset

```bash
# Downloads CIFAR-10 to data/cifar-10-batches-py/
bash experiments/simple_triton_resnet/download_data.sh
```

### 4. Run Inference Client

```bash
# Create virtual environment
uv venv --python 3.10 experiments/simple_triton_resnet/.venv
source experiments/simple_triton_resnet/.venv/bin/activate

# Install dependencies
uv pip install -r experiments/simple_triton_resnet/requirements.txt

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
├── README.md              # This file
├── client.py              # Triton HTTP inference client
├── requirements.txt       # Python dependencies (tritonclient, pillow, numpy)
├── setup_model.sh         # Downloads ResNet50 and generates config.pbtxt
├── download_data.sh       # Downloads CIFAR-10 dataset
├── run_smoke_test.sh      # Automated end-to-end test
└── .venv/                 # Virtual environment (created by smoke test)
```

## Suggested Improvements

### Organization
1. **Create `scripts/` subdirectory:**
   - Move `setup_model.sh`, `download_data.sh`, `run_smoke_test.sh` → `scripts/`
   - Keep `client.py` and `requirements.txt` at root
   - Cleaner separation between automation scripts and client code

2. **Add `.gitignore` for experiment:**
   ```
   .venv/
   *.pyc
   __pycache__/
   .pytest_cache/
   ```

3. **Create `tests/` directory:**
   - Add unit tests for preprocessing functions
   - Add integration test fixture with mock responses
   - Separate test data from production scripts

### Code Quality
1. **Extract preprocessing to separate module:**
   - `preprocessing.py` with `resize_and_normalize()` function
   - Reusable across different experiments
   - Easier to unit test

2. **Add configuration file:**
   - `config.yaml` with server URL, model name, tensor names
   - Avoid hardcoded values in client.py
   - Environment-specific configs (dev/staging/prod)

3. **Add logging:**
   - Use Python `logging` instead of `print()`
   - Configurable log levels (DEBUG/INFO/WARNING)
   - Log to file for debugging

### Proposed Structure
```
experiments/simple_triton_resnet/
├── README.md
├── client.py              # Main inference client
├── requirements.txt
├── config.yaml           # NEW: Configuration
├── preprocessing.py      # NEW: Shared preprocessing functions
├── .gitignore           # NEW: Ignore venv and cache
├── scripts/             # NEW: Automation scripts
│   ├── setup_model.sh
│   ├── download_data.sh
│   └── run_smoke_test.sh
└── tests/               # NEW: Test suite
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_client.py
```

### Additional Enhancements
1. **Add performance metrics:**
   - Measure inference latency
   - Track throughput (requests/sec)
   - Log to CSV for analysis

2. **Support batch inference:**
   - Modify client to send multiple images
   - Compare single vs batch performance

3. **Add model comparison:**
   - Download different ResNet variants (18, 34, 101)
   - Compare accuracy/speed tradeoffs
   - Automate model benchmarking

4. **CI/CD integration:**
   - GitHub Actions workflow to run smoke test
   - Fail PR if Triton inference breaks
   - Cache Docker images for faster builds

#!/usr/bin/env python3
"""
Triton ResNet50 Inference Client with built-in setup and smoke test.

Modes:
  python client.py --index 0              # Run inference
  python client.py --setup                # Setup model and data
  python client.py --smoke-test           # Full end-to-end test
"""
import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


def get_repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def setup_model():
    """Download ResNet50 model using triton/add_dummy_model.sh."""
    print("==> Setting up ResNet50 model...")
    repo_root = get_repo_root()
    script = repo_root / "triton" / "add_dummy_model.sh"
    
    if not script.exists():
        raise FileNotFoundError(f"Model setup script not found: {script}")
    
    subprocess.run(["bash", str(script), "resnet50", "1"], cwd=script.parent, check=True)
    print("==> Model setup complete!")


def download_data():
    """Download CIFAR-10 dataset."""
    print("==> Downloading CIFAR-10 dataset...")
    repo_root = get_repo_root()
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    cifar_dir = data_dir / "cifar-10-batches-py"
    if (cifar_dir / "data_batch_1").exists():
        print(f"CIFAR-10 already exists at {cifar_dir}")
        return
    
    # Download
    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_tar = data_dir / "cifar-10-python.tar.gz"
    
    print(f"Downloading from {cifar_url}...")
    subprocess.run(["curl", "-L", "--fail", "-o", str(cifar_tar), cifar_url], check=True)
    
    print("Extracting...")
    subprocess.run(["tar", "-xzf", str(cifar_tar), "-C", str(data_dir)], check=True)
    cifar_tar.unlink()
    
    print(f"==> CIFAR-10 downloaded to {cifar_dir}")


def start_triton():
    """Start Triton server using docker compose."""
    print("==> Starting Triton server...")
    repo_root = get_repo_root()
    
    # Stop existing
    subprocess.run(
        ["docker", "compose", "--profile", "triton", "down"],
        cwd=repo_root,
        capture_output=True
    )
    
    # Start new
    subprocess.run(
        ["docker", "compose", "--profile", "triton", "up", "-d", "--build"],
        cwd=repo_root,
        check=True
    )


def wait_for_triton(timeout: int = 60):
    """Wait for Triton to become ready."""
    print("==> Waiting for Triton to be ready...")
    url = "http://localhost:8000/v2/health/ready"
    
    for i in range(1, timeout + 1):
        try:
            result = subprocess.run(
                ["curl", "-sf", url],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print("Triton ready!")
                return
        except subprocess.TimeoutExpired:
            pass
        
        if i < timeout:
            time.sleep(1)
    
    print(f"Triton not ready after {timeout}s. Printing logs...")
    repo_root = get_repo_root()
    subprocess.run(
        ["docker", "compose", "--profile", "triton", "logs", "triton"],
        cwd=repo_root
    )
    raise RuntimeError("Triton failed to start")


def load_cifar10_image(dataset_dir: Path, index: int) -> Tuple[np.ndarray, int]:
    """Load a CIFAR-10 image."""
    batch_path = dataset_dir / "data_batch_1"
    if not batch_path.exists():
        raise FileNotFoundError(f"Could not find {batch_path}. Run with --setup first.")
    
    with open(batch_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    
    images = data["data"]
    labels = data["labels"]
    
    if not (0 <= index < len(images)):
        raise IndexError(f"Index {index} out of range (0-{len(images)-1})")
    
    img_flat = images[index]
    label = labels[index]
    
    # Reshape from flat to 32x32 RGB
    r = img_flat[0:1024].reshape(32, 32)
    g = img_flat[1024:2048].reshape(32, 32)
    b = img_flat[2048:3072].reshape(32, 32)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    
    return img, label


def preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocess image for ResNet50 (ImageNet normalization)."""
    # Resize to 224x224
    pil = Image.fromarray(img)
    pil = pil.resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    
    # ImageNet mean/std normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    
    # Convert to NCHW
    chw = np.transpose(arr, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0)
    
    return nchw


def run_inference(
    server_url: str,
    model: str,
    dataset_dir: Path,
    index: int,
    input_name: str,
    output_name: str
):
    """Run inference on a CIFAR-10 image."""
    # Load and preprocess
    img, _ = load_cifar10_image(dataset_dir, index)
    tensor = preprocess(img)
    
    # Inference
    try:
        client = httpclient.InferenceServerClient(url=server_url, verbose=False)
        
        infer_input = httpclient.InferInput(input_name, tensor.shape, "FP32")
        infer_input.set_data_from_numpy(tensor, binary_data=True)
        
        infer_output = httpclient.InferRequestedOutput(output_name, binary_data=True)
        
        response = client.infer(model_name=model, inputs=[infer_input], outputs=[infer_output])
        output_data = response.as_numpy(output_name)
        
        if output_data is None:
            raise RuntimeError("No output data returned from Triton")
        
        # Get top-5 predictions
        logits = output_data[0]
        top5 = np.argsort(-logits)[:5]
        
        print(f"Top-5 class indices: {top5.tolist()}")
        print("Note: CIFAR-10 labels don't match ImageNet classes; this is a smoke test.")
        
    except InferenceServerException as e:
        raise SystemExit(f"Triton inference failed: {e}")


def smoke_test():
    """Run full end-to-end smoke test."""
    print("=== Starting Smoke Test ===\n")
    
    # Setup
    setup_model()
    download_data()
    
    # Start server
    start_triton()
    wait_for_triton()
    
    # Run inference
    repo_root = get_repo_root()
    dataset_dir = repo_root / "data" / "cifar-10-batches-py"
    
    print("\n==> Running inference...")
    run_inference(
        server_url="localhost:8000",
        model="resnet50",
        dataset_dir=dataset_dir,
        index=0,
        input_name="gpu_0/data_0",
        output_name="gpu_0/softmax_1"
    )
    
    print("\n=== Smoke Test Complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="Triton ResNet50 client with setup and smoke test"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--setup", action="store_true", help="Setup model and data only")
    mode_group.add_argument("--smoke-test", action="store_true", help="Run full smoke test")
    
    # Inference options
    parser.add_argument("--server-url", default="localhost:8000", help="Triton server URL")
    parser.add_argument("--model", default="resnet50", help="Model name")
    parser.add_argument("--dataset-dir", help="Dataset directory (default: data/cifar-10-batches-py)")
    parser.add_argument("--index", type=int, default=0, help="CIFAR-10 image index (0-9999)")
    parser.add_argument("--input-name", default="gpu_0/data_0", help="Model input tensor name")
    parser.add_argument("--output-name", default="gpu_0/softmax_1", help="Model output tensor name")
    
    args = parser.parse_args()
    
    # Execute mode
    if args.setup:
        setup_model()
        download_data()
    elif args.smoke_test:
        smoke_test()
    else:
        # Run inference
        repo_root = get_repo_root()
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else repo_root / "data" / "cifar-10-batches-py"
        
        if not dataset_dir.exists():
            print(f"Dataset not found at {dataset_dir}")
            print("Run with --setup to download it")
            sys.exit(1)
        
        run_inference(
            server_url=args.server_url,
            model=args.model,
            dataset_dir=dataset_dir,
            index=args.index,
            input_name=args.input_name,
            output_name=args.output_name
        )


if __name__ == "__main__":
    main()

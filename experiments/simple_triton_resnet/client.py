#!/usr/bin/env python3
"""
Minimal Triton HTTP client to run inference on ResNet50 using a CIFAR-10 image.
- Uses Triton HTTP/REST V2 API via requests (no tritonclient dependency)
- Preprocess: resize to 224x224, NCHW, float32, ImageNet mean/std normalization
- Input name: gpu_0/data_0
- Output name: gpu_0/resnet_node_output_0
"""
import argparse
import json
import os
import pickle
import sys
from typing import Tuple

import numpy as np
from PIL import Image
import requests


def load_cifar10_image(dataset_dir: str, index: int) -> Tuple[np.ndarray, int]:
    """Load one CIFAR-10 image (returns HWC RGB uint8 and label)."""
    batch_path = os.path.join(dataset_dir, "data_batch_1")
    if not os.path.isfile(batch_path):
        raise FileNotFoundError(f"Could not find {batch_path}. Run download_data.sh first.")
    with open(batch_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    images = data["data"]  # shape (10000, 3072)
    labels = data["labels"]
    if not (0 <= index < len(images)):
        raise IndexError(f"Index {index} out of range for CIFAR batch (size {len(images)}).")
    img_flat = images[index]
    label = labels[index]
    # CIFAR stores in CHW order per channel stacked (R 1024, G 1024, B 1024)
    r = img_flat[0:1024].reshape(32, 32)
    g = img_flat[1024:2048].reshape(32, 32)
    b = img_flat[2048:3072].reshape(32, 32)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)  # HWC
    return img, label


def preprocess(img: np.ndarray) -> np.ndarray:
    """Resize to 224x224, normalize to ImageNet stats, and convert to NCHW float32 batch size 1."""
    pil = Image.fromarray(img)
    pil = pil.resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32) / 255.0  # HWC, range [0,1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # HWC -> CHW
    chw = np.transpose(arr, (0, 1, 2))
    chw = np.transpose(chw, (2, 0, 1))
    # NCHW
    nchw = np.expand_dims(chw, axis=0)
    return nchw


def infer_http(server_url: str, model_name: str, input_name: str, output_name: str, tensor: np.ndarray):
    """Send HTTP/JSON inference request to Triton and return outputs."""
    url = server_url.rstrip("/") + f"/v2/models/{model_name}/infer"
    payload = {
        "inputs": [
            {
                "name": input_name,
                "shape": list(tensor.shape),
                "datatype": "FP32",
                "data": tensor.flatten().tolist(),
            }
        ],
        "outputs": [
            {"name": output_name}
        ],
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Inference failed: HTTP {resp.status_code} {resp.text}")
    out = resp.json()
    outputs = out.get("outputs", [])
    if not outputs:
        raise RuntimeError(f"No outputs in response: {out}")
    data = outputs[0].get("data")
    if data is None:
        raise RuntimeError(f"Output missing data field: {outputs[0]}")
    arr = np.array(data, dtype=np.float32)
    # Model outputs [1, 1000]
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="http://localhost:8000", help="Triton HTTP endpoint")
    p.add_argument("--model", default="resnet50", help="Model name in Triton")
    p.add_argument("--dataset-dir", default="data/cifar-10-batches-py", help="Dataset folder under repo root data/")
    p.add_argument("--index", type=int, default=0, help="CIFAR image index (0-9999)")
    p.add_argument("--input-name", default="gpu_0/data_0", help="Model input name")
    p.add_argument("--output-name", default="gpu_0/resnet_node_output_0", help="Model output name")
    args = p.parse_args()

    # Resolve dataset dir relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(repo_root, dataset_dir)

    img, label = load_cifar10_image(dataset_dir, args.index)
    tensor = preprocess(img)
    logits = infer_http(args.server_url, args.model, args.input_name, args.output_name, tensor)
    # Top-5 indices
    top5 = np.argsort(-logits)[:5]
    print("Top-5 class indices:", top5.tolist())
    print("Note: CIFAR-10 labels don't match ImageNet classes; this is a smoke test.")


if __name__ == "__main__":
    main()

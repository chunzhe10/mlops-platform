#!/usr/bin/env python3
"""
Triton HTTP client (official tritonclient) to run inference on ResNet50 using a CIFAR-10 image.
- Uses tritonclient[http]
- Preprocess: resize to 224x224, NCHW, float32, ImageNet mean/std normalization
- Input name: gpu_0/data_0
- Output name: gpu_0/softmax_1
- Model has max_batch_size: 0 (no batching), expects shape [1, 3, 224, 224]
"""
import argparse
import os
import pickle
from typing import Tuple

import numpy as np
from PIL import Image

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


def load_cifar10_image(dataset_dir: str, index: int) -> Tuple[np.ndarray, int]:
    batch_path = os.path.join(dataset_dir, "data_batch_1")
    if not os.path.isfile(batch_path):
        raise FileNotFoundError(f"Could not find {batch_path}. Run download_data.sh first.")
    with open(batch_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    images = data["data"]
    labels = data["labels"]
    if not (0 <= index < len(images)):
        raise IndexError(f"Index {index} out of range for CIFAR batch (size {len(images)}).")
    img_flat = images[index]
    label = labels[index]
    r = img_flat[0:1024].reshape(32, 32)
    g = img_flat[1024:2048].reshape(32, 32)
    b = img_flat[2048:3072].reshape(32, 32)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return img, label


def preprocess(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = pil.resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0)
    return nchw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="localhost:8000", help="Triton HTTP endpoint host:port")
    p.add_argument("--model", default="resnet50", help="Model name in Triton")
    p.add_argument("--dataset-dir", default="data/cifar-10-batches-py", help="Dataset folder under repo root data/")
    p.add_argument("--index", type=int, default=0, help="CIFAR image index (0-9999)")
    p.add_argument("--input-name", default="gpu_0/data_0", help="Model input name")
    p.add_argument("--output-name", default="gpu_0/softmax_1", help="Model output name")
    args = p.parse_args()

    # Resolve dataset dir relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(repo_root, dataset_dir)

    img, _ = load_cifar10_image(dataset_dir, args.index)
    tensor = preprocess(img)

    try:
        client = httpclient.InferenceServerClient(url=args.server_url, verbose=False)
        # Prepare inputs/outputs
        infer_input = httpclient.InferInput(args.input_name, tensor.shape, "FP32")
        infer_input.set_data_from_numpy(tensor, binary_data=True)
        infer_output = httpclient.InferRequestedOutput(args.output_name, binary_data=True)

        response = client.infer(model_name=args.model, inputs=[infer_input], outputs=[infer_output])
        output_data = response.as_numpy(args.output_name)
        if output_data is None:
            raise RuntimeError("No output data returned from Triton.")
        # output_data shape expected [1, 1000]
        logits = output_data[0]
        top5 = np.argsort(-logits)[:5]
        print("Top-5 class indices:", top5.tolist())
        print("Note: CIFAR-10 labels don't match ImageNet classes; this is a smoke test.")
    except InferenceServerException as e:
        raise SystemExit(f"Triton inference failed: {e}")


if __name__ == "__main__":
    main()

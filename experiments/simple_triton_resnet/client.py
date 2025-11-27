#!/usr/bin/env python3
"""
Triton ResNet50 Inference Client.

Usage:
  python client.py --index 0              # Run inference on CIFAR-10 image
  python client.py --index 123            # Different image
  python client.py --image path/to/img    # Custom image file
"""
import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class ImagePreprocessor:
    """Preprocess images for ResNet50 inference."""
    
    # ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    TARGET_SIZE = (224, 224)
    
    @classmethod
    def preprocess(cls, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ResNet50 inference.
        
        Args:
            img: Input image as numpy array (HWC, RGB, uint8)
            
        Returns:
            Preprocessed tensor (1CHW, FP32, normalized)
        """
        # Resize to 224x224
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(cls.TARGET_SIZE, Image.BILINEAR)
        
        # Convert to float32 and normalize to [0, 1]
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        arr = (arr - cls.IMAGENET_MEAN) / cls.IMAGENET_STD
        
        # Convert HWC to CHW
        chw = np.transpose(arr, (2, 0, 1))
        
        # Add batch dimension: CHW -> 1CHW
        nchw = np.expand_dims(chw, axis=0)
        
        return nchw


class CIFAR10Dataset:
    """Load images from CIFAR-10 dataset."""
    
    def __init__(self, dataset_dir: Path):
        """
        Initialize CIFAR-10 dataset loader.
        
        Args:
            dataset_dir: Directory containing CIFAR-10 batches
        """
        self.dataset_dir = dataset_dir
        
    def load_image(self, index: int, batch_num: int = 1) -> Tuple[np.ndarray, int]:
        """
        Load a CIFAR-10 image.
        
        Args:
            index: Image index within batch (0-9999)
            batch_num: Batch number (1-5)
            
        Returns:
            Tuple of (image array HWC RGB uint8, label)
            
        Raises:
            FileNotFoundError: If dataset not found
            IndexError: If index out of range
        """
        batch_path = self.dataset_dir / f"data_batch_{batch_num}"
        if not batch_path.exists():
            raise FileNotFoundError(
                f"Could not find {batch_path}. "
                "Run 'python setup.py --data' to download."
            )
        
        with open(batch_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        
        images = data["data"]
        labels = data["labels"]
        
        if not (0 <= index < len(images)):
            raise IndexError(f"Index {index} out of range (0-{len(images)-1})")
        
        # Reshape from flat to 32x32 RGB
        img_flat = images[index]
        r = img_flat[0:1024].reshape(32, 32)
        g = img_flat[1024:2048].reshape(32, 32)
        b = img_flat[2048:3072].reshape(32, 32)
        img = np.stack([r, g, b], axis=-1).astype(np.uint8)
        
        label = labels[index]
        
        return img, label


class TritonInferenceClient:
    """Client for Triton inference server."""
    
    def __init__(
        self,
        server_url: str = "localhost:8000",
        model_name: str = "resnet50",
        input_name: str = "gpu_0/data_0",
        output_name: str = "gpu_0/softmax_1",
        verbose: bool = False
    ):
        """
        Initialize Triton inference client.
        
        Args:
            server_url: Triton server URL (host:port)
            model_name: Model name to query
            input_name: Input tensor name
            output_name: Output tensor name
            verbose: Enable verbose logging
        """
        self.server_url = server_url
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.verbose = verbose
        
        self._client = httpclient.InferenceServerClient(
            url=server_url,
            verbose=verbose
        )
        
    def predict(self, image: np.ndarray, top_k: int = 5) -> List[int]:
        """
        Run inference on preprocessed image.
        
        Args:
            image: Preprocessed image tensor (1CHW, FP32)
            top_k: Number of top predictions to return
            
        Returns:
            List of top-k class indices
            
        Raises:
            InferenceServerException: If inference fails
        """
        try:
            # Prepare input
            infer_input = httpclient.InferInput(
                self.input_name,
                image.shape,
                "FP32"
            )
            infer_input.set_data_from_numpy(image, binary_data=True)
            
            # Prepare output
            infer_output = httpclient.InferRequestedOutput(
                self.output_name,
                binary_data=True
            )
            
            # Run inference
            response = self._client.infer(
                model_name=self.model_name,
                inputs=[infer_input],
                outputs=[infer_output]
            )
            
            # Extract predictions
            output_data = response.as_numpy(self.output_name)
            if output_data is None:
                raise RuntimeError("No output data returned from Triton")
            
            # Get top-k predictions
            logits = output_data[0]
            top_indices = np.argsort(-logits)[:top_k]
            
            return top_indices.tolist()
            
        except InferenceServerException as e:
            raise SystemExit(f"Triton inference failed: {e}")
            
    def predict_from_cifar10(
        self,
        dataset_dir: Path,
        index: int = 0,
        top_k: int = 5
    ) -> List[int]:
        """
        Run inference on CIFAR-10 image.
        
        Args:
            dataset_dir: CIFAR-10 dataset directory
            index: Image index (0-9999)
            top_k: Number of top predictions
            
        Returns:
            List of top-k class indices
        """
        # Load image
        dataset = CIFAR10Dataset(dataset_dir)
        img, label = dataset.load_image(index)
        
        # Preprocess
        preprocessor = ImagePreprocessor()
        tensor = preprocessor.preprocess(img)
        
        # Predict
        return self.predict(tensor, top_k=top_k)
        
    def predict_from_file(
        self,
        image_path: Path,
        top_k: int = 5
    ) -> List[int]:
        """
        Run inference on image file.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions
            
        Returns:
            List of top-k class indices
        """
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_array = np.asarray(img)
        
        # Preprocess
        preprocessor = ImagePreprocessor()
        tensor = preprocessor.preprocess(img_array)
        
        # Predict
        return self.predict(tensor, top_k=top_k)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Triton ResNet50 inference client"
    )
    
    # Server configuration
    parser.add_argument(
        "--server-url",
        default="localhost:8000",
        help="Triton server URL (default: localhost:8000)"
    )
    parser.add_argument(
        "--model",
        default="resnet50",
        help="Model name (default: resnet50)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--index",
        type=int,
        default=0,
        help="CIFAR-10 image index (0-9999, default: 0)"
    )
    input_group.add_argument(
        "--image",
        type=Path,
        help="Path to custom image file"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="CIFAR-10 dataset directory (default: data/cifar-10-batches-py)"
    )
    
    # Model tensor names
    parser.add_argument(
        "--input-name",
        default="gpu_0/data_0",
        help="Input tensor name (default: gpu_0/data_0)"
    )
    parser.add_argument(
        "--output-name",
        default="gpu_0/softmax_1",
        help="Output tensor name (default: gpu_0/softmax_1)"
    )
    
    # Output
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = TritonInferenceClient(
        server_url=args.server_url,
        model_name=args.model,
        input_name=args.input_name,
        output_name=args.output_name,
        verbose=args.verbose
    )
    
    # Run inference
    try:
        if args.image:
            # Custom image file
            if not args.image.exists():
                print(f"Error: Image file not found: {args.image}")
                sys.exit(1)
            predictions = client.predict_from_file(args.image, top_k=args.top_k)
            print(f"Top-{args.top_k} predictions for {args.image}:")
        else:
            # CIFAR-10 dataset
            repo_root = Path(__file__).resolve().parent.parent.parent
            dataset_dir = args.dataset_dir or repo_root / "data" / "cifar-10-batches-py"
            
            if not dataset_dir.exists():
                print(f"Error: Dataset not found at {dataset_dir}")
                print("Run 'python setup.py --data' to download it")
                sys.exit(1)
            
            predictions = client.predict_from_cifar10(
                dataset_dir,
                index=args.index,
                top_k=args.top_k
            )
            print(f"Top-{args.top_k} predictions for CIFAR-10 image {args.index}:")
        
        print(predictions)
        
        if not args.image:
            print("\nNote: CIFAR-10 labels don't match ImageNet classes.")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

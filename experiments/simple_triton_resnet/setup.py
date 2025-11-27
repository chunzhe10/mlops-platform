#!/usr/bin/env python3
"""
Setup and infrastructure management for Triton ResNet50 experiment.

Usage:
  python setup.py --model           # Download ResNet50 model
  python setup.py --data            # Download CIFAR-10 dataset
  python setup.py --all             # Setup everything
  python setup.py --start-triton    # Start Triton server
  python setup.py --smoke-test      # Full end-to-end test
"""
import argparse
import subprocess
import time
from pathlib import Path
from typing import Optional


class TritonSetup:
    """Manage Triton model and data setup."""
    
    def __init__(self, repo_root: Optional[Path] = None):
        """
        Initialize setup manager.
        
        Args:
            repo_root: Repository root directory. Auto-detected if not provided.
        """
        if repo_root is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
        self.repo_root = repo_root
        self.data_dir = repo_root / "data"
        self.triton_dir = repo_root / "triton"
        
    def setup_model(self, model_name: str = "resnet50", version: str = "1"):
        """
        Download and configure model for Triton.
        
        Args:
            model_name: Model name to setup
            version: Model version
        """
        print(f"==> Setting up {model_name} model (version {version})...")
        script = self.triton_dir / "add_dummy_model.sh"
        
        if not script.exists():
            raise FileNotFoundError(f"Model setup script not found: {script}")
        
        subprocess.run(
            ["bash", str(script), model_name, version],
            cwd=script.parent,
            check=True
        )
        print("==> Model setup complete!")
        
    def download_data(self):
        """Download CIFAR-10 dataset."""
        print("==> Downloading CIFAR-10 dataset...")
        self.data_dir.mkdir(exist_ok=True)
        
        cifar_dir = self.data_dir / "cifar-10-batches-py"
        if (cifar_dir / "data_batch_1").exists():
            print(f"CIFAR-10 already exists at {cifar_dir}")
            return
        
        # Download
        cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        cifar_tar = self.data_dir / "cifar-10-python.tar.gz"
        
        print(f"Downloading from {cifar_url}...")
        subprocess.run(
            ["curl", "-L", "--fail", "-o", str(cifar_tar), cifar_url],
            check=True
        )
        
        print("Extracting...")
        subprocess.run(
            ["tar", "-xzf", str(cifar_tar), "-C", str(self.data_dir)],
            check=True
        )
        cifar_tar.unlink()
        
        print(f"==> CIFAR-10 downloaded to {cifar_dir}")


class TritonServer:
    """Manage Triton inference server lifecycle."""
    
    def __init__(self, repo_root: Optional[Path] = None):
        """
        Initialize Triton server manager.
        
        Args:
            repo_root: Repository root directory. Auto-detected if not provided.
        """
        if repo_root is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
        self.repo_root = repo_root
        self.health_url = "http://localhost:8000/v2/health/ready"
        
    def start(self, rebuild: bool = True):
        """
        Start Triton server using docker compose.
        
        Args:
            rebuild: Whether to rebuild the Docker image
        """
        print("==> Starting Triton server...")
        
        # Stop existing
        subprocess.run(
            ["docker", "compose", "--profile", "triton", "down"],
            cwd=self.repo_root,
            capture_output=True
        )
        
        # Start new
        cmd = ["docker", "compose", "--profile", "triton", "up", "-d"]
        if rebuild:
            cmd.append("--build")
            
        subprocess.run(cmd, cwd=self.repo_root, check=True)
        
    def stop(self):
        """Stop Triton server."""
        print("==> Stopping Triton server...")
        subprocess.run(
            ["docker", "compose", "--profile", "triton", "down"],
            cwd=self.repo_root,
            check=True
        )
        
    def wait_until_ready(self, timeout: int = 60):
        """
        Wait for Triton to become ready.
        
        Args:
            timeout: Maximum seconds to wait
            
        Raises:
            RuntimeError: If Triton doesn't become ready within timeout
        """
        print("==> Waiting for Triton to be ready...")
        
        for i in range(1, timeout + 1):
            try:
                result = subprocess.run(
                    ["curl", "-sf", self.health_url],
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
        
        # Timeout - print logs and fail
        print(f"Triton not ready after {timeout}s. Printing logs...")
        subprocess.run(
            ["docker", "compose", "--profile", "triton", "logs", "triton"],
            cwd=self.repo_root
        )
        raise RuntimeError("Triton failed to start")
        
    def get_logs(self):
        """Get Triton container logs."""
        result = subprocess.run(
            ["docker", "compose", "--profile", "triton", "logs", "triton"],
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )
        return result.stdout


class SmokeTest:
    """End-to-end smoke test orchestrator."""
    
    def __init__(self):
        """Initialize smoke test."""
        self.setup = TritonSetup()
        self.server = TritonServer()
        
    def run(self):
        """Run full smoke test."""
        print("=== Starting Smoke Test ===\n")
        
        # Setup
        self.setup.setup_model()
        self.setup.download_data()
        
        # Start server
        self.server.start()
        self.server.wait_until_ready()
        
        # Run inference
        print("\n==> Running inference...")
        from client import TritonInferenceClient
        
        client = TritonInferenceClient(
            server_url="localhost:8000",
            model_name="resnet50",
            input_name="gpu_0/data_0",
            output_name="gpu_0/softmax_1"
        )
        
        dataset_dir = self.setup.repo_root / "data" / "cifar-10-batches-py"
        predictions = client.predict_from_cifar10(dataset_dir, index=0)
        
        print(f"Top-5 class indices: {predictions}")
        print("Note: CIFAR-10 labels don't match ImageNet classes; this is a smoke test.")
        
        print("\n=== Smoke Test Complete ===")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup and manage Triton ResNet50 experiment"
    )
    
    # Setup options
    parser.add_argument("--model", action="store_true", help="Setup model")
    parser.add_argument("--data", action="store_true", help="Download dataset")
    parser.add_argument("--all", action="store_true", help="Setup everything")
    
    # Server management
    parser.add_argument("--start-triton", action="store_true", help="Start Triton server")
    parser.add_argument("--stop-triton", action="store_true", help="Stop Triton server")
    parser.add_argument("--no-rebuild", action="store_true", help="Don't rebuild Docker image")
    
    # Testing
    parser.add_argument("--smoke-test", action="store_true", help="Run full smoke test")
    
    args = parser.parse_args()
    
    # Execute
    if args.smoke_test:
        test = SmokeTest()
        test.run()
    elif args.all:
        setup = TritonSetup()
        setup.setup_model()
        setup.download_data()
    elif args.model:
        setup = TritonSetup()
        setup.setup_model()
    elif args.data:
        setup = TritonSetup()
        setup.download_data()
    elif args.start_triton:
        server = TritonServer()
        server.start(rebuild=not args.no_rebuild)
        server.wait_until_ready()
    elif args.stop_triton:
        server = TritonServer()
        server.stop()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

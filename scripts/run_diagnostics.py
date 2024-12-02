import sys

import torch


def run_diagnostics():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available")

    # Test PyTorch CUDA access
    if torch.cuda.is_available():
        try:
            x = torch.rand(5, 3)
            print("CPU Tensor:", x)
            x = x.cuda()
            print("GPU Tensor:", x)
            print("CUDA access successful")
        except Exception as e:
            print("Error accessing CUDA:", str(e))


if __name__ == "__main__":
    run_diagnostics()

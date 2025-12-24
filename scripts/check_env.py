#!/usr/bin/env python3
import torch
import sys
import os

def check_env():
    print("--- Environment Check ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check MPS (Metal Performance Shaders) for M1/M2
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available (Mac M1/M2): {mps_available}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    # Check models directory
    models_dir = "models"
    if os.path.exists(models_dir):
        print(f"Models directory found: {os.listdir(models_dir)}")
    else:
        print("Warning: 'models' directory not found.")

    # Check CosyVoice
    if os.path.exists("CosyVoice"):
        print("CosyVoice source found.")
    else:
        print("Warning: 'CosyVoice' source not found. Please clone it.")

if __name__ == "__main__":
    check_env()

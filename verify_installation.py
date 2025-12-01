#!/usr/bin/env python3
"""
Installation Verification Script for RAFTcorr
=============================================
Quick verification that all components are properly installed.
"""

import sys
import importlib

def check_module(name, description):
    """Check if a module can be imported."""
    try:
        importlib.import_module(name)
        print(f"[OK] {description}")
        return True
    except ImportError as e:
        print(f"[FAIL] {description} - {e}")
        return False

def check_cuda():
    """Check CUDA availability - REQUIRED for RAFT-DIC."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"[OK] CUDA available - {torch.cuda.device_count()} GPU(s)")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 4.0:
                print("[WARN] GPU has less than 4GB memory - may struggle with large images")
                
        else:
            print("[FATAL] CUDA not available!")
            print("RAFT-DIC REQUIRES NVIDIA GPU with CUDA support.")
            print("Please install NVIDIA GPU driver and CUDA toolkit.")
            
        return cuda_available
    except Exception as e:
        print(f"[FAIL] CUDA check failed - {e}")
        return False

def check_cuda_extension():
    """Check if custom CUDA extension is available."""
    try:
        import alt_cuda_corr
        print("[OK] Custom CUDA correlation extension available")
        return True
    except ImportError:
        print("[WARN] Custom CUDA extension not available (will use PyTorch built-in)")
        return False

def main():
    """Run all verification checks."""
    print("RAFTcorr Installation Verification")
    print("=" * 50)
    
    # Core dependencies
    checks = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"), 
        ("matplotlib", "Matplotlib"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("tifffile", "TiffFile"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("customtkinter", "CustomTkinter"),
    ]
    
    failed = 0
    for module, desc in checks:
        if not check_module(module, desc):
            failed += 1
    
    print("\nGPU Support:")
    print("-" * 20)
    cuda_ok = check_cuda()
    check_cuda_extension()
    
    print(f"\nPython version: {sys.version}")
    
    if failed > 0:
        print(f"\n[ERROR] {failed} dependencies missing. Please run: pip install -e .")
        return 1
        
    if not cuda_ok:
        print(f"\n[FATAL] CUDA GPU support is REQUIRED for RAFT-DIC!")
        print("This application cannot run without NVIDIA GPU and CUDA.")
        print("\nRequired setup:")
        print("1. NVIDIA GPU (RTX series recommended)")
        print("2. Updated GPU driver")
        print("3. CUDA Toolkit 11.8+ or 12.x")
        print("4. Verify with: nvidia-smi")
        return 1
    
    print(f"\n[SUCCESS] All dependencies installed and GPU verified!")
    print("[READY] RAFTcorr is ready to run with GPU acceleration!")
    
    return 0

if __name__ == "__main__":
    exit(main())

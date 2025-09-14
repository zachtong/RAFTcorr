#!/usr/bin/env python3
"""
2D-RAFT-DIC-GUI Setup Script
============================
A clean, efficient setup for RAFT-DIC with automatic CUDA detection.

Authors: Zixiang (Zach) Tong @ UT-Austin, Lehu Bu @ UT-Austin
License: MIT
"""

import subprocess
import sys
import os
import re
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext

# Try to import torch for CUDA extensions
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def read_readme():
    """Read README.md for package description."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "A graphical user interface for Digital Image Correlation using RAFT neural network."

def detect_cuda_version():
    """Detect CUDA version from nvidia-smi or nvcc."""
    cuda_version = None
    
    # Try nvidia-smi first (more reliable)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Extract CUDA version from nvidia-smi output
            match = re.search(r'CUDA Version: (\d+)\.(\d+)', result.stdout)
            if match:
                major, minor = match.groups()
                cuda_version = f"{major}.{minor}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fallback to nvcc
    if not cuda_version:
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                match = re.search(r'release (\d+)\.(\d+)', result.stdout)
                if match:
                    major, minor = match.groups()
                    cuda_version = f"{major}.{minor}"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    return cuda_version


def install_pytorch_with_cuda():
    """Install PyTorch with CUDA support - CUDA is REQUIRED."""
    print("\n" + "="*60)
    print("RAFT-DIC REQUIRES CUDA GPU SUPPORT")
    print("="*60)
    
    cuda_version = detect_cuda_version()
    if not cuda_version:
        print("[FATAL] CUDA not detected!")
        print("RAFT-DIC requires NVIDIA GPU with CUDA support.")
        print("\nPlease install:")
        print("1. NVIDIA GPU driver")
        print("2. CUDA Toolkit 11.8+ or 12.x")
        print("3. Verify with: nvidia-smi")
        raise RuntimeError("CUDA not available - RAFT-DIC cannot run without GPU")
    
    print(f"[OK] Detected CUDA version: {cuda_version}")
    
    # Only install CUDA-enabled PyTorch
    major_version = int(cuda_version.split('.')[0])
    if major_version >= 12:
        index_url = "https://download.pytorch.org/whl/cu124"  # CUDA 12.x uses cu124
    else:
        index_url = "https://download.pytorch.org/whl/cu118"  # CUDA 11.x uses cu118
    
    print(f"Installing CUDA-enabled PyTorch from: {index_url}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch>=2.0.0", "torchvision>=0.15.0", "torchaudio>=2.0.0",
            "--index-url", index_url
        ], timeout=300)
        
        print("[OK] PyTorch installation completed!")
        
        # MANDATORY GPU verification
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        
        if not torch.cuda.is_available():
            print("[FATAL] PyTorch installed but CUDA not available!")
            print("This usually means:")
            print("1. GPU driver is outdated")
            print("2. CUDA runtime mismatch")
            print("3. PyTorch CPU version was installed by mistake")
            raise RuntimeError("CUDA verification failed - RAFT-DIC requires GPU")
        
        print(f"[OK] CUDA available: {torch.version.cuda}")
        print(f"[OK] GPU count: {torch.cuda.device_count()}")
        print(f"[OK] GPU device: {torch.cuda.get_device_name(0)}")
        
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[OK] GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 4.0:
            print("[WARN] GPU has less than 4GB memory - may struggle with large images")
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PyTorch installation failed: {e}")
        print(f"\nManual installation command:")
        print(f"pip install torch torchvision torchaudio --index-url {index_url}")
        raise
    except subprocess.TimeoutExpired:
        print("[ERROR] PyTorch installation timed out")
        raise

def get_cuda_extensions():
    """Get CUDA extensions - REQUIRED for optimal performance."""
    extensions = []
    
    if not os.path.exists('alt_cuda_corr'):
        print("[WARN] CUDA correlation source not found - will use slower PyTorch built-in")
        return extensions
    
    if not TORCH_AVAILABLE:
        print("[WARN] PyTorch not available during setup - CUDA extension will be skipped")
        return extensions
        
    try:
        import torch
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available - CUDA extension will be skipped")
            return extensions
            
        extensions.append(
            CUDAExtension(
                name='alt_cuda_corr',
                sources=[
                    'alt_cuda_corr/correlation.cpp',
                    'alt_cuda_corr/correlation_kernel.cu'
                ],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3', '--use_fast_math', '--expt-relaxed-constexpr']
                }
            )
        )
        print("[OK] CUDA correlation extension will be built for optimal performance")
        
    except Exception as e:
        print(f"[WARN] CUDA extension setup failed: {e}")
        print("The application will still work but may be slower")
    
    return extensions

class CustomInstallCommand(install):
    """Custom installation command."""
    def run(self):
        install_pytorch_with_cuda()
        install.run(self)

class CustomDevelopCommand(develop):
    """Custom development installation command."""
    def run(self):
        install_pytorch_with_cuda()
        develop.run(self)

# Core dependencies (excluding PyTorch which is handled separately)
REQUIREMENTS = [
    "numpy>=1.21.0",
    "scipy>=1.7.0", 
    "matplotlib>=3.5.0",
    "opencv-python>=4.5.0",
    "Pillow>=8.0.0",
    "tifffile>=2021.7.2",
]

# Get CUDA extensions
cuda_extensions = get_cuda_extensions()

setup(
    name="raft-dic-gui",
    version="1.0.0",
    author="Zixiang (Zach) Tong, Lehu Bu",
    author_email="zachtong@utexas.edu",
    description="Digital Image Correlation using RAFT neural network with GPU acceleration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zachtong/2D-RAFT-DIC-GUI",
    
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires=">=3.8",
    
    # CUDA extensions
    ext_modules=cuda_extensions,
    
    entry_points={
        "console_scripts": [
            "raft-dic-gui=main_GUI:main",
            "raft-dic-demo=demo:main",
        ],
    },
    
    package_data={
        "": ["*.yml", "*.yaml", "*.md", "*.txt", "*.pth"],
        "models": ["*.pth"],
        "examples": ["**/*"],
        "alt_cuda_corr": ["*.cu", "*.cpp"],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    
    keywords="digital-image-correlation DIC RAFT neural-network CUDA GPU",
    zip_safe=False,
    
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
        'build_ext': BuildExtension.with_options(use_ninja=False) if TORCH_AVAILABLE else build_ext,
    },
)
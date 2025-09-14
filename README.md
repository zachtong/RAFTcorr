# RAFT-DIC-GUI

**Digital Image Correlation using RAFT Neural Network with GPU acceleration**
![Demo video](assets/RAFT_DIC_demo.gif)


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A powerful GUI tool combining RAFT (Recurrent All-Pairs Field Transforms) neural network with Digital Image Correlation for precise displacement field analysis.

## Features

- 🚀 **GPU-accelerated processing** with automatic CUDA detection
- 🖼️ **Interactive ROI selection** and real-time preview
- 🔄 **Dual analysis modes**: accumulative and incremental DIC
- 📊 **Advanced visualization** with U/V displacement components
- 🎬 **Animation playback** of displacement sequences
- 🔧 **Gaussian smoothing** for noise reduction
- 💾 **Multiple export formats**: NPY, MAT files

## Quick Start

### Prerequisites (MANDATORY)

⚠️ **RAFT-DIC REQUIRES NVIDIA GPU - CPU-ONLY WILL NOT WORK** ⚠️

- **Python 3.8+**
- **NVIDIA GPU** with CUDA support (RTX 2060 or newer recommended)
- **4GB+ VRAM** (8GB+ recommended for large images)
- **CUDA Toolkit 11.8+ or 12.x**
- **Updated NVIDIA GPU driver**

### Pre-Installation Check

**BEFORE installing, verify your GPU setup:**

```bash
nvidia-smi  # Should show your GPU and CUDA version
```

If `nvidia-smi` fails, install:
1. [NVIDIA GPU Driver](https://www.nvidia.com/drivers/)
2. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Installation (One Command)

```bash
git clone https://github.com/zachtong/2D-RAFT-DIC-GUI.git
cd 2D-RAFT-DIC-GUI
pip install -e .
```

The setup script will:
- ✅ **Verify CUDA is available** (FAILS if no GPU)
- ✅ Auto-detect your CUDA version
- ✅ Install compatible PyTorch with CUDA support  
- ✅ Install all dependencies
- ✅ Build optimized CUDA extensions
- ✅ Verify GPU functionality and memory

### Verify Installation

```bash
python verify_installation.py
```

Expected output (with GPU):
```
[OK] NumPy
[OK] PyTorch  
[OK] CUDA available - 1 GPU(s)
  CUDA version: 12.1
  GPU: NVIDIA GeForce RTX 4070
  GPU memory: 12.0 GB
[SUCCESS] All dependencies installed and GPU verified!
[READY] RAFT-DIC is ready to run with GPU acceleration!
```

**If you see CUDA errors, the installation will FAIL - this is intentional!**

### Launch Application

```bash
raft-dic-gui
```

Or directly:
```bash
python main_GUI.py
```

## Usage Workflow

1. **Select Images**: Choose input image directory
2. **Set Output**: Specify results directory  
3. **Define ROI**: Draw region of interest on reference image
4. **Configure Parameters**:
   - Processing mode (accumulative/incremental)
   - Crop size and stride
   - Smoothing options
5. **Run Analysis**: Process and view results

## Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **Crop Size** | Processing window size (H×W) | 256×256, 512×512 |
| **Stride** | Step between windows | 64, 128 |
| **Max Displacement** | Expected displacement range | 10-50 pixels |
| **Smoothing Sigma** | Gaussian filter strength | 0.5-2.0 |

## Output

- **NPY files**: Raw displacement arrays
- **MAT files**: MATLAB-compatible format
- **Visualizations**: Displacement field plots
- **Window layouts**: Processing grid visualization

## Troubleshooting

### CUDA Not Detected
```bash
# Check if CUDA is properly installed
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Common issues:**
1. **"CUDA not detected"** - Install NVIDIA driver + CUDA toolkit
2. **"PyTorch CPU version"** - Uninstall and reinstall with CUDA
3. **"GPU memory insufficient"** - Close other GPU applications

### Manual PyTorch Installation (CUDA only)
```bash
# For CUDA 12.x (recommended)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip uninstall torch torchvision torchaudio  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### No GPU? This Software Won't Work
RAFT-DIC is computationally intensive and **requires GPU acceleration**. There is no CPU fallback mode.

### Performance Tips
- Use crop sizes that are multiples of 32
- Reduce stride for higher resolution (slower processing)
- Enable smoothing for noisy images
- Monitor GPU memory usage with large images

## Model Setup (IMPORTANT)

The pre-trained RAFT model (`raft-dic_v1.pth`) should be automatically included. If missing:
1. Download from releases
2. Place in `models/` directory

## Contributing

We welcome contributions! Please:
- Follow the existing code style
- Add tests for new features
- Update documentation

## Citation

If you use this software in research:

```bibtex
[TBD]
```

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## Authors

- **Zixiang (Zach) Tong** - University of Texas at Austin
- **Lehu Bu** - University of Texas at Austin

## Acknowledgments

- Original RAFT implementation: [Princeton Vision Lab](https://github.com/princeton-vl/RAFT)
- Digital Image Correlation community
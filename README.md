# SNN Temporal Fusion — Single GPU

A CUDA-accelerated training framework for Spiking Neural Networks (SNNs) using temporal fusion, based on the paper:

> **Towards Scalable GPU-Accelerated SNN Training via Temporal Fusion**
> Yanchen Li, Jiachun Li, Kebin Sun, Luziwei Leng, Ran Cheng
> Southern University of Science and Technology
> arXiv:2408.00280v1 [cs.AI] — August 2024

This repository implements and extends the single-GPU temporal fusion method with bug fixes, a Ternary LIF neuron variant, and training/evaluation scripts for static and event-based datasets.

---

## What is Temporal Fusion?

Standard SNN training processes time steps sequentially — each step requires a separate GPU kernel launch, a memory read, and a memory write. This incurs linearly scaling overhead with time step count.

Temporal fusion reorders computation to a **temporal-major** order: all time steps for a given layer are fused into a single GPU kernel call. Each neuron is assigned its own GPU thread, and memory operations across all time steps are merged, eliminating repeated read/write overhead.

Benchmarked against existing SNN libraries on NVIDIA A100 GPUs, temporal fusion achieves **5× to 40× acceleration** depending on time step count.

---

## Project Structure

```
.
├── kernel/
│   ├── include/
│   │   ├── lif_kernel.h
│   │   └── ternary_lif_kernel.h
│   ├── src/
│   │   ├── neuron.cpp          # PyTorch/pybind11 bindings
│   │   ├── lif_kernel.cu       # CUDA kernel — standard LIF
│   │   └── ternary_lif_kernel.cu  # CUDA kernel — Ternary LIF
│   └── setup.py
├── neuron/
│   ├── __init__.py
│   ├── lif.py                  # LIF autograd function + nn.Module
│   └── ternary_lif.py          # TernaryLIF autograd function + nn.Module
├── fused_resnet.py             # Spiking-ResNet18/34/50 definitions
├── single_gpu_test.py          # Training + evaluation script
└── install.sh                  # Full environment setup script
```

---

## Neuron Models

### LIF (Leaky Integrate-and-Fire)
Standard binary spiking neuron. Forward dynamics:

```
v(t) = kτ · v(t-1) · (1 - y(t-1)) + Vrest · y(t-1) + x(t)
y(t) = 1 if v(t) >= Vth, else 0
```

### TernaryLIF
Extended variant with three output states: {-1, 0, 1}. Fires positively when membrane potential exceeds threshold, negatively when it falls below negative threshold:

```
y(t) =  1  if v(t) >= Vth
y(t) = -1  if v(t) <= -Vth
y(t) =  0  otherwise
```

Both use a hard sigmoid surrogate gradient for backpropagation.

---

## Requirements

- Linux (Ubuntu 20.04+)
- NVIDIA GPU (sm_80+ recommended; tested on RTX 3050)
- CUDA Toolkit 12.x
- Python 3.12
- PyTorch with matching CUDA wheel
- ninja (for fast kernel compilation)

---

## Installation

### Automated

```bash
chmod +x install.sh
./install.sh
```

The script creates a virtualenv, installs PyTorch, SpikingJelly, ninja, and builds the CUDA kernel.

### Manual

```bash
# Create and activate venv
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install spikingjelly ninja matplotlib

# Build CUDA kernel
cd kernel/
python setup.py build_ext --inplace
cd ..
```

---

## Usage

```bash
python single_gpu_test.py \
    --device 0 \
    --dataset MNIST \
    --neuron LIF \
    --arch Spiking-ResNet18
```

### Arguments

| Argument | Options | Default |
|---|---|---|
| `--device` | GPU index | `0` |
| `--data_root` | Path to datasets | `../data` |
| `--dataset` | `MNIST`, `CIFAR-10`, `N-MNIST`, `DvsGesture` | `MNIST` |
| `--neuron` | `LIF`, `TernaryLIF` | `LIF` |
| `--arch` | `Spiking-ResNet18`, `Spiking-ResNet34`, `Spiking-ResNet50` | `Spiking-ResNet18` |
| `--timing` | Enable epoch timing | `True` |

### Data directory setup

For event-based datasets (N-MNIST, DvsGesture), create the data directory first:

```bash
mkdir -p ./data/NMNIST
mkdir -p ./data/DvsGesture
```

Then pass `--data_root ./data`. SpikingJelly will download automatically.

---

## Results

Tested on RTX 3050 (4GB VRAM), T=32 time steps, 5 epochs, Adam lr=1e-3.

### MNIST — Spiking-ResNet18 + LIF

| Epoch | Train Time (s) | Test Time (s) | Accuracy (%) |
|---|---|---|---|
| 0 | 185.9 | 15.0 | 91.78 |
| 1 | 187.8 | 15.0 | 94.64 |
| 2 | 190.1 | 15.7 | 97.48 |
| 3 | 191.0 | 15.3 | 97.33 |
| 4 | 190.4 | 15.2 | 97.26 |

Training and accuracy curves are saved to `results.png` after each run.

---

## Architecture

### Spiking-ResNet variants

| Model | Block | Layers | Parameters |
|---|---|---|---|
| Spiking-ResNet18 | BasicBlock | [2,2,2,2] | ~11M |
| Spiking-ResNet34 | BasicBlock | [3,4,6,3] | ~21M |
| Spiking-ResNet50 | Bottleneck (expansion=4) | [3,4,6,3] | ~25M |

ANN operators (Conv, BN) run on merged `[T*B, C, H, W]` tensors. Spiking neuron layers run on split `[T, B, C, H, W]` tensors via the fused CUDA kernel. Output logits are averaged across time steps before loss computation.

---

## Bug Fixes Applied

The following bugs were identified and fixed during development:

| File | Issue | Fix |
|---|---|---|
| `lif_kernel.cu` | `auto` function pointer rejected by nvcc | Explicit `if/else` branch per kernel |
| `lif_kernel.cu` | `fabs` instead of `fabsf` | `fabsf` for device float code |
| `lif_kernel.cu` | `sigmoidSurrogate` missing `- threshold` shift | `coshf(alpha * (x - threshold))` |
| `lif_kernel.cu` | `__device__` functions declared after kernels | Moved above first kernel |
| `lif_kernel.cu` | Integer literals `0`, `1`, `0.5` without `f` suffix | `0.0f`, `1.0f`, `0.5f` |
| `lif_kernel.cu` | `gridSize * blockSize` signed overflow vs `size_t` | Cast to `size_t` before compare |
| `lif_kernel.h` | `size_t` used without `#include <cstddef>` | Added include |
| `ternary_lif_kernel.h` | Same as above | Added include |
| `lif.py` | `FusedLIF.backward` returned 5 gradients, forward had 6 inputs | Added `None` for `use_tv` |
| `ternary_lif.py` | Same gradient count mismatch | Added `None` for `use_tv` |
| `fused_resnet.py` | `Bottleneck` set `identity = x` before downsample check | Moved inside `if/else` |
| `single_gpu_test.py` | `logits.mean(0)` called twice (model already does it) | Removed redundant call |
| `single_gpu_test.py` | One-hot labels passed to `CrossEntropyLoss` | Integer labels passed directly |
| `ternary_lif.py` | `import kernel import temporal_fusion_kernal` | `from kernel import temporal_fusion_kernel` |

---

## Known Limitations

- CIFAR-10 accuracy is near random chance (~10%) with default hyperparameters. Requires lower lr (`1e-4`), more time steps (T=64+), and data augmentation to train effectively.
- Multi-node scaling not implemented — single node only.
- BF16/FP16 kernel templating not yet applied — kernels run in FP32 only.

---

## Citation

```bibtex
@article{li2024temporal,
  title   = {Towards Scalable GPU-Accelerated SNN Training via Temporal Fusion},
  author  = {Li, Yanchen and Li, Jiachun and Sun, Kebin and Leng, Luziwei and Cheng, Ran},
  journal = {arXiv preprint arXiv:2408.00280},
  year    = {2024}
}
```

## Acknowledgements

SNN architectures adapted from [SpikingJelly](https://github.com/fangwei123456/spikingjelly):

> Fang, W., Chen, Y., Ding, J., Yu, Z., Masquelier, T., Chen, D., Huang, L., Zhou, H., Li, G., Tian, Y.
> SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence.
> Science Advances, vol. 9, no. 40, eadi1480, 2023.

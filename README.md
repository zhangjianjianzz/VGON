# VGON Environment Setup

VGON scripts here require Python version 3.11 and above.

This guide provides instructions for setting up the VGON (Variational Generative Optimization Network) environment using either `uv` or `pip`.

## Prerequisites

- Python 3.11 or higher
- Git

## Method 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager and project manager, whose installation is simple as [guide](https://docs.astral.sh/uv/#installation)


### Setup with uv
1. Clone the repository to local disk.
    ```bash
    git clone https://github.com/zhangjianjianzz/VGON.git && cd VGON
    ```
2. Create virtual environment if it wasn't contained
    ```bash
    uv venv .venv
    ```
    The python virtual environment will be located at `.venv` fold under root path in default. Linux/macOS users could activate it in shell by typing
    ```bash
    source .venv/bin/activate
    ```
3. Install dependecies in `.venv` for VGON scripts 
    ```bash
    uv sync --extra full
    ```
    `plot` and `notebook` groups are optional for VGON itself. 

4. Running experiments with uv
    ```bash
    # Run specific experiments
    uv run python BP/HXXZ/vgon_xxz.py
    uv run python Gap/Mix.py
    uv run python Degeneracy/H232/H232.py

    # Or activate environment first
    source .venv/bin/activate
    python BP/HXXZ/plot.py
    ```

## Method 2: Using pip

### Setup with pip

```bash
# Clone the repository
git clone <your-repo-url>
cd VGON

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support (CUDA 12.1), use instead:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the package with dependencies
pip install -e .

# Install optional dependencies
pip install -e ".[plot]"     # For plotting
pip install -e ".[dev]"      # For development
pip install -e ".[gpu]"      # For GPU acceleration
pip install -e ".[all]"      # Install everything

# Manual installation of ptitprince (if needed)
pip install git+https://github.com/pog87/PtitPrince.git
```

## Verify Installation

Test your installation by running a simple example:

```python
import torch
import pennylane as qml
import numpy as np

# Test basic functionality
print(f"PyTorch version: {torch.__version__}")
print(f"PennyLane version: {qml.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test quantum device
dev = qml.device("default.qubit", wires=2)
print("✓ PennyLane quantum device created successfully")
```

## Project Structure

```
VGON/
├── BP/              # Barren Plateau experiments
│   ├── HXXZ/        # Heisenberg XXZ model
│   └── Z1Z2/        # Z1Z2 model
├── Degeneracy/      # Degeneracy detection experiments
│   ├── H232/        # H232 Hamiltonian
│   └── MG/          # Graph states
├── Gap/             # Nonlocality gap experiments
└── results/         # Experimental results
```

## Running Experiments

### Barren Plateau Experiments
```bash
# HXXZ model
python BP/HXXZ/vgon_xxz.py    # Run VGON training
python BP/HXXZ/vqe_xxz.py     # Run VQE baseline
python BP/HXXZ/plot.py        # Generate plots

# Z1Z2 model
python BP/Z1Z2/vgon_z1z2.py
python BP/Z1Z2/vqe_z1z2.py
python BP/Z1Z2/plot.py
```

### Degeneracy Detection
```bash
# H232 Hamiltonian
python Degeneracy/H232/H232.py
python Degeneracy/H232/plot.py

# Graph states
python Degeneracy/MG/MG.py
python Degeneracy/MG/plot.py
```

### Nonlocality Gap
```bash
python Gap/Mix.py              # Train gap model
matlab -batch "run('Gap/plotGap.m')"  # Generate MATLAB plots
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: If you have a GPU but CUDA is not detected, reinstall PyTorch with CUDA support
2. **PennyLane device errors**: Ensure you have the correct PennyLane plugins installed
3. **Memory issues**: Reduce batch sizes in the configuration files
4. **Import errors**: Make sure all dependencies are installed correctly

### GPU Setup

For GPU acceleration:


### Development Setup

For contributors:

## Support

- Paper: [Commun Phys 8, 334 (2025)](https://doi.org/10.1038/s42005-025-02261-4)
- Issues: Create an issue on the GitHub repository
- Documentation: See individual module docstrings and comments

# [Development Containers](https://containers.dev/)
DevContainers' configuration in vscode/GitHub Codespace.


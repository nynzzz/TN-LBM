#!/bin/bash
# venv_setup.sh - Setup virtual environment for TN-LBM project
# Usage: ./venv_setup.sh

set -e  # Exit on error

VENV_NAME="tn-lbm-venv"
PYTHON_VERSION="3.12"

echo "=== TN-LBM Virtual Environment Setup ==="
echo ""

# Find Python 3.12
if command -v python${PYTHON_VERSION} &> /dev/null; then
    PYTHON_CMD="python${PYTHON_VERSION}"
elif command -v python3 &> /dev/null && python3 --version | grep -q "3.12"; then
    PYTHON_CMD="python3"
else
    echo "ERROR: Python ${PYTHON_VERSION} not found."
    echo "Please install Python ${PYTHON_VERSION}:"
    echo "  macOS:   brew install python@${PYTHON_VERSION}"
    echo "  Ubuntu:  sudo apt install python${PYTHON_VERSION} python${PYTHON_VERSION}-venv"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"
echo ""

# Remove existing venv if present
if [ -d "$VENV_NAME" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_NAME"
fi

# Create virtual environment
echo "Creating virtual environment: $VENV_NAME"
$PYTHON_CMD -m venv "$VENV_NAME"

# Activate venv
source "$VENV_NAME/bin/activate"

echo "Virtual environment activated"
echo "Python location: $(which python)"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install base scientific packages
echo "Installing base packages (numpy, scipy, matplotlib, pandas)..."
pip install numpy scipy matplotlib pandas --quiet

# Install numba (with pre-built wheels to avoid LLVM issues)
echo "Installing numba (pre-built wheels)..."
pip install --only-binary :all: llvmlite numba --quiet

# Install quimb tensor network library
echo "Installing quimb..."
pip install quimb --quiet

# Install numpy-hilbert-curve for 2D->1D mappings
echo "Installing numpy-hilbert-curve..."
pip install numpy-hilbert-curve --quiet

# Install Jupyter
echo "Installing Jupyter..."
pip install jupyter ipykernel --quiet

# Register Jupyter kernel
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=tn-lbm --display-name="TN-LBM (Python ${PYTHON_VERSION})" --quiet

echo ""
echo "=== Installation Complete ==="
echo ""

# Verify installation
echo "Verifying installation..."
python -c "
import numpy as np
import scipy
import matplotlib
import numba
import quimb as qu

print(f'  numpy:      {np.__version__}')
print(f'  scipy:      {scipy.__version__}')
print(f'  matplotlib: {matplotlib.__version__}')
print(f'  numba:      {numba.__version__}')
print(f'  quimb:      {qu.__version__}')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source $VENV_NAME/bin/activate"
echo ""

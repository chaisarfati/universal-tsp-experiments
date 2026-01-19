#!/bin/bash

# -----------------------------------------------------------------------------
# Minimal & safe launch script (macOS)
# Adds only a virtual environment to avoid NumPy / pip issues
# -----------------------------------------------------------------------------

PYTHON=python

# 1. Check python
if ! command -v $PYTHON &>/dev/null; then
    echo "[ERROR] Python is not installed."
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON --version)"

# 2. Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    $PYTHON -m venv .venv
fi

# 3. Activate venv
source .venv/bin/activate

# 4. Ensure pip is available
if ! python -m pip --version &>/dev/null; then
    echo "[ERROR] pip is not available in this Python."
    exit 1
fi

# 5. Install dependencies
cd tsp_analysis
echo "[INFO] Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd ..

# 6. Launch the viewer
echo "[INFO] Launching the viewer..."
python -m tsp_analysis.viewer_tk

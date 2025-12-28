#!/bin/bash
# -----------------------------------------------------------------------------
# Universal launch script for the TSP viewer (macOS / Linux)
# -----------------------------------------------------------------------------
# 1. Move to the script directory

# 2. Detect python3.9 or python3
if command -v python3.9 &>/dev/null; then
    PYTHON=python3.9
elif command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "[ERROR] Python 3 is not installed. Please install it using 'brew install python'."
    exit 1
fi

# 3. Check for pip
if ! $PYTHON -m pip --version &>/dev/null; then
    echo "[ERROR] pip is not installed. Installing now..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON get-pip.py
    rm get-pip.py
fi

# 4. Install dependencies
cd tsp_analysis
echo "[INFO] Installing dependencies..."
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt
cd ..

# 5. Launch the viewer
echo "[INFO] Launching the viewer..."
$PYTHON -m tsp_analysis.viewer_tk

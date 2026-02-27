#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$BASE_DIR"

PYTHON=python3
VENV_DIR="$BASE_DIR/.venv"

# 1. Clean up broken venv if it exists but activation fails
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[INFO] Cleaning up broken virtual environment..."
    rm -rf "$VENV_DIR"
fi

# 2. Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# 3. Activate and Verify
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[ERROR] Failed to activate virtual environment. Is python3-venv installed?"
    exit 1
fi

# 4. Install dependencies using the venv's pip specifically
echo "[INFO] Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$BASE_DIR/tsp_analysis/requirements.txt"

# 5. Launch
echo "[INFO] Launching the viewer..."
"$VENV_DIR/bin/python" -m tsp_analysis.viewer_tk

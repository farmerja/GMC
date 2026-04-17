#!/bin/bash
# Activation script for GMC virtual environment

echo "Activating GMC virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo ""
echo "Installed packages:"
echo "  - numpy"
echo "  - torch"
echo "  - numba"
echo "  - matplotlib"
echo "  - flow-matching"
echo "  - scipy"
echo "  - tqdm"
echo ""
echo "To run multigrid_steady_2d.py:"
echo "  python physics/multigrid_steady_2d.py"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"

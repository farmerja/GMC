# Virtual Environment Setup

A Python virtual environment has been created for the GMC project with all required dependencies.

## Quick Start

### Activate the virtual environment:

```bash
source venv/bin/activate
```

Or use the activation script:

```bash
source activate_venv.sh
```

### Deactivate the virtual environment:

```bash
deactivate
```

## Installed Dependencies

The following packages have been installed:

- **numpy** (2.4.4) - Numerical computing
- **torch** (2.11.0) - PyTorch for deep learning
- **numba** (0.65.0) - JIT compilation for Python
- **matplotlib** (3.10.8) - Plotting and visualization
- **flow-matching** (1.0.10) - Flow matching for generative models
- **scipy** (1.17.1) - Scientific computing
- **tqdm** (4.67.3) - Progress bars
- And their dependencies...

See `requirements.txt` for the complete list.

## Running the Code

Once the virtual environment is activated, you can run the multigrid code:

```bash
python physics/multigrid_steady_2d.py
```

Or any other Python scripts in the project.

## Reinstalling Dependencies

If you need to reinstall dependencies in a fresh virtual environment:

```bash
pip install -r requirements.txt
```

## Notes

- The virtual environment is located in the `venv/` directory
- Python version: 3.12
- The environment includes PyTorch for CPU/Apple Silicon

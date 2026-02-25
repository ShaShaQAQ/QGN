# QGN Analyzer

Numerical toolkit for Quantum Geometric Nesting (QGN) in flat-band tight-binding models, following [PhysRevX.14.041004](https://doi.org/10.1103/PhysRevX.14.041004).

## Structure

```
qgn/          # Core Python module
  core.py     # Nesting operator, nestability map, nesting matrix
  models.py   # SSH, custom TB, TBG Wannier Hamiltonians
  geometry.py # Band structure, Berry curvature, Chern number, quantum distance
tests/        # pytest test suite
notebooks/    # Jupyter analysis notebooks
  01_SSH.ipynb
  02_custom_TB.ipynb
  03_TBG.ipynb
docs/         # Generated figures
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
pytest tests/          # run tests
jupyter notebook       # open notebooks
```

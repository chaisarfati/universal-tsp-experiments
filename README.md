# Universal TSP Analysis Tool

A Python framework for experimenting with **linear ordering heuristics** for the Traveling Salesman Problem (TSP) on planar point sets, and comparing them against near-optimal TSP tours.

The project provides tools for generating point sets, applying ordering heuristics, computing TSP solutions, and visualizing results interactively.

---

## Features

- Generate structured point sets on grids
- Apply linear ordering heuristics (e.g. Hilbert, Z-order)
- Compute near-optimal TSP tours using the LKH heuristic
- Compare heuristic paths against TSP tours
- Detect and visualize backtracking structures
- Interactive GUI for exploration and experimentation

---

## Running the Viewer

To launch the interactive visualization interface:

```bash
bash launch_viewer.sh
```

The viewer allows you to:
 - browse saved experiments
 - select heuristics and grid sizes
 - compare heuristic paths with TSP tours
 - inspect ordering ranks interactively
 - recompute paths after modifying point sets

## Adding a New Ordering Heuristic

Ordering heuristics are defined in heuristics.py.

A heuristic is a function mapping a set of 2D points to a linear order:

```python
def my_heuristic(points: np.ndarray, **optional_parameters):
    ...
    return indices, codes
```

### Requirements
 - points: array of shape (N, 2)
 - indices: permutation of {0, ..., N-1}
 - codes: one-dimensional code per point
 - register the heuristic in heuristics_registry:
    heuristics_registry = {
        "Hilbert": hilbert_order,
        "Z-order": zcurve_order,
        "Platzman": platzman_order,
        "MyHeuristic": my_heuristic,
    }

## Dependencies

The code was compiled with Python 3.9.
All dependencies are listed in requirements.txt.

## LKH Solver Note

The TSP baseline is computed using the Lin–Kernighan–Helsgaun (LKH) heuristic via the lk_heuristic library.

This project relies on a modified fork of lk_heuristic where the solve(...) function returns the best_tour, which is required for analysis and visualization.
See commit:
https://github.com/chaisarfati/lk_heuristic/commit/6613e01d115939698b9860dfef0d451981a5c040
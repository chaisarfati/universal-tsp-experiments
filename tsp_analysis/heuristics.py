import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
import zCurve as z
from .geometry import sierpinski_curve

# -------------------------
# HILBERT ORDER
# -------------------------
def hilbert_order(points: np.ndarray, p: int = 10):
    """
    Order points by their distance along a 2D Hilbert curve of order p.
    Returns (indices, codes).
    """
    n = 2
    hilbert_curve = HilbertCurve(p, n)
    R = 2 ** p

    def point_to_index(pt):
        x_int = min(R - 1, max(0, int(pt[0] * R)))
        y_int = min(R - 1, max(0, int(pt[1] * R)))
        return hilbert_curve.distance_from_point([x_int, y_int])

    hilbert_indices = [point_to_index(pt) for pt in points]
    return np.argsort(hilbert_indices), hilbert_indices


# -------------------------
# Z-CURVE (MORTON ORDER)
# -------------------------
def zcurve_order(points: np.ndarray, bits: int = 16):
    """
    Order points by Z-curve (Morton code).
    Returns (indices, morton_codes).
    """
    R = 2 ** bits

    def float_to_int_coords(points, bits=16):
        return [
            (min(R - 1, max(0, int(x * R))),
             min(R - 1, max(0, int(y * R))))
            for x, y in points
        ]

    int_points = float_to_int_coords(points, bits)
    morton_codes = z.par_interlace(int_points, dims=2, bits_per_dim=bits)
    return np.argsort(morton_codes), morton_codes


# -------------------------
# PLATZMAN–BARTHOLDI (APPROX. VIA SIERPIŃSKI)
# -------------------------
def platzman_order(points: np.ndarray):
    """
    Approximate Platzman–Bartholdi order by mapping to the closest point
    on a Sierpiński curve support and sorting by that position.
    Returns (indices, sfc_codes).
    """
    N = len(points)
    if N == 0:
        return np.array([]), np.array([])

    # Minimal iterations to cover N support points (heuristic)
    iteration = int(np.ceil(0.5 * np.log2(N)))
    sfc_points = sierpinski_curve(iteration)

    # Vectorized nearest-neighbor to Sierpiński support points
    diff = points[:, None, :] - sfc_points[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    nearest_idx = np.argmin(dist2, axis=1)

    sfc_codes = nearest_idx
    indices = np.argsort(sfc_codes)
    return indices, sfc_codes

# === Accessible heuristics ===
heuristics_registry_dict = {
    "Hilbert": hilbert_order,
    "Z-order": zcurve_order,
    "Platzman": platzman_order,
}
import numpy as np
from math import cos, sin, pi, log2
import math
import random

# -------------------------
# 1. GRID GENERATION
# -------------------------
def generate_grid_points(M: int) -> np.ndarray:
    """Generate a uniform MxM grid of points in [0,1]^2 (cell centers)."""
    step = 1 / M
    half_step = step / 2
    points = []
    for i in range(M):
        for j in range(M):
            x = i * step + half_step
            y = j * step + half_step
            points.append(np.array([x, y]))
    return np.array(points)


# -------------------------
# 2. BASIC GEOMETRIC HELPERS
# -------------------------
def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Midpoint between two 2D points."""
    return np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])

def triangle_centroid(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Centroid of a triangle."""
    return np.array([(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3])

def generate_random_global_line():
    theta = np.random.uniform(0, math.pi)
    d = np.array([math.cos(theta), math.sin(theta)])
    n = np.array([-d[1], d[0]])
    p0 = np.random.uniform(0, 1, size=2)
    return p0, d, n, theta

def point_distance_to_line(x, p0, n):
    return abs(np.dot(n, x - p0))

# -------------------------
# 3. SIERPIŃSKI CURVE SUPPORT
# -------------------------
def sierpinski_sub(tri, iters: int):
    """Recursive subdivision of a triangle into two sub-triangles, returning centroids."""
    p1, p2, p3 = tri
    if iters == 0:
        return [triangle_centroid(p1, p2, p3)]
    else:
        mid = midpoint(p1, p3)
        sub1 = [p1, mid, p2]
        sub2 = [p2, mid, p3]
        return sierpinski_sub(sub1, iters - 1) + sierpinski_sub(sub2, iters - 1)

def sierpinski_curve(iterations: int = 3, size: float = 1.0) -> np.ndarray:
    """Build Sierpiński (triangular) curve support points (approximate)."""
    len_side = size
    offset = np.array([0.0, 0.0])

    tri1 = [offset + [0, len_side], offset + [0, 0], offset + [len_side, 0]]
    tri2 = [offset + [len_side, 0], offset + [len_side, len_side], offset + [0, len_side]]

    half1 = sierpinski_sub(tri1, iterations)
    half2 = sierpinski_sub(tri2, iterations)
    points = np.array(half1 + half2)
    return points

# -------------------------
# 4. DYADIC SQUARES SUPPORT
# -------------------------
def generate_dyadic_squares(scale: int):
    """Returns all dyadic squares of scale t as [((x0,y0), size), ...]."""
    step = 1 / (2 ** scale)
    squares = []
    for i in range(2 ** scale):
        for j in range(2 ** scale):
            x0 = i * step
            y0 = j * step
            squares.append(((x0, y0), step))  # bottom-left corner and size
    return squares



def filter_points_in_square(points, x0, y0, size):
    """
    Return a boolean mask selecting points inside the dyadic square
    [x0, x0 + size) x [y0, y0 + size), using half-open intervals to avoid
    ambiguities on square boundaries.
    """
    x1, y1 = x0 + size, y0 + size
    pts = np.asarray(points, float)
    mask = (
        (pts[:, 0] >= x0) & (pts[:, 0] < x1) &
        (pts[:, 1] >= y0) & (pts[:, 1] < y1)
    )
    return mask


# -------------------------
# 4. COSMASIAN GEOMETRY
# -------------------------
def find_backtrack_state(points, order, l=0.3, w=0.08, angle_steps=180, max_tries=50):
    """
    Detect a backtracking configuration along a given linear order.

    This function acts as an oracle for identifying backtracking states
    in a linearly ordered point set. It randomly selects pivot points and
    scans a discrete set of directions to search for a local geometric
    configuration witnessing a backtrack, as defined by Cosmas Kravaris 
    in https://www.arxiv.org/pdf/2412.16448

    A backtracking state consists of a pivot point p and two subsequent
    points q1 and q2 (appearing later in the linear order) such that q1
    lies in a backward slab and q2 lies in a forward slab of length l and
    width w, relative to a chosen direction. The configuration additionally
    enforces that q1 precedes q2 in the global order.

    Parameters
    ----------
    points : array-like of shape (N, 2)
        Set of planar points.
    order : array-like of shape (N,)
        Linear order of the points, given as a permutation of indices.
    l : float, optional
        Half-length of the forward and backward slabs along the direction
        vector (default: 0.3).
    w : float, optional
        Width of the slabs orthogonal to the direction vector (default: 0.08).
    angle_steps : int, optional
        Number of discrete directions uniformly sampled in [0, π) for the
        directional scan (default: 180).
    max_tries : int, optional
        Maximum number of random pivot points tested before giving up
        (default: 50).

    Returns
    -------
    state : dict or None
        If a backtracking configuration is found, returns a dictionary
        containing:
            - 'p'        : the pivot point,
            - 'p_index'  : index of p in the linear order,
            - 'theta'    : direction angle,
            - 'd'        : unit direction vector,
            - 'n'        : orthogonal unit vector,
            - 'R1'       : list of points in the backward slab,
            - 'R2'       : list of points in the forward slab,
            - 'q1'       : selected backward point,
            - 'q2'       : selected forward point.
        If no configuration is detected, returns None.

    Notes
    -----
    This procedure is heuristic and randomized. It does not guarantee
    detection of all possible backtracking configurations, but it is
    designed to reliably identify such structures in practice. For this
    reason, it is treated as a trusted oracle within the experimental
    pipeline.
    """
    points = np.array(points, float)
    ordered = [points[i] for i in order]

    if len(ordered) < 3:
        return None

    # position in the linear order
    pos = {id(q): i for i, q in enumerate(ordered)}
    
    angles = np.linspace(0, pi, angle_steps, endpoint=False)

    for _ in range(max_tries):
        # pivot aléatoire
        p_index = random.randint(2, len(ordered) - 3)
        p = ordered[p_index]

        # scan des angles
        for theta in angles:
            d = np.array([cos(theta), sin(theta)])
            d /= np.linalg.norm(d)
            n = np.array([-d[1], d[0]])

            R1, R2 = [], []

            for q in ordered[p_index + 1:]:
                v = q - p
                tproj = np.dot(v, d)
                uproj = np.dot(v, n)

                if abs(uproj) > w / 2:
                    continue

                if -l <= tproj < 0:
                    R1.append(q)
                elif 0 < tproj <= l:
                    R2.append(q)

            valid_pairs = [
                (r1, r2)
                for r1 in R1
                for r2 in R2
                if pos[id(r1)] < pos[id(r2)]
            ]

            if valid_pairs:
                q1, q2 = random.choice(valid_pairs)
                return {
                    "p": p, # the pivot point in the backtrack
                    "p_index": p_index, # the pivot's index in the order
                    "theta": theta, # the angle of the line L_0
                    "d": d, # vector of line L_0
                    "n": n, # vector normal (perpendicular) to d (used to calculate distance from a point)
                    "R1": R1, # rectangular area containting backtrack q1 or q2
                    "R2": R2, # rectangular area containting backtrack q1 or q2
                    "q1": q1, # q1 succed pivot in order but precedes q2
                    "q2": q2, # q2 succeeds pivot and q2
                }


    return None

def scale_lw_with_t(t, alpha=0.6, beta=0.2):
    s = 2 ** (-t)
    return alpha * s, beta * s

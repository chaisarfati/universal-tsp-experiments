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


def find_backtrack_state_kravaris(points, order, l=0.3, w=0.08, angle_steps=180, max_tries=50):
    """
    Find a Kravaris-style "backtrack state" inside a *fixed* linear order.

    This is a *randomized heuristic oracle* (not an exhaustive decision procedure).
    It attempts to witness a backtracking configuration as used in Cosmas Kravaris'
    lower-bound framework (see: https://www.arxiv.org/pdf/2412.16448).

    Informal definition of the searched configuration
    -----------------------------------------------
    We look for:
        - a pivot point p (picked from the order),
        - a direction d (unit vector, discretized by an angle grid),
        - two points q1, q2 that appear *after* p in the given order,

    such that q1 and q2 fall into two thin directional slabs (rectangles) anchored at p:

        R1 (backward slab):  tproj in [-l, 0)   and |uproj| <= w/2
        R2 (forward slab):   tproj in (0,  l]   and |uproj| <= w/2

    where:
        v     = q - p
        tproj = <v, d>   (projection along direction d)
        uproj = <v, n>   (projection along perpendicular n)

    Additionally we enforce an order-consistency constraint:
        q1 must appear before q2 in the *global* order (among the ordered list),
    i.e. pos(q1) < pos(q2). This matches the intended “backtrack” witness where
    the tour (induced by the linear order) goes forward/backward in a structured way.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        Input planar points in [0,1]^2 (or any planar domain).
    order : array-like, shape (N,)
        A permutation (or subsequence) of indices specifying the linear order.
        IMPORTANT: This function never changes `order`; it only searches within it.
    l : float
        Half-length of the slabs along the direction d (same l for forward/backward).
        If you run inside a dyadic square of side `size`, you typically want l <= O(size).
    w : float
        Full width of slabs orthogonal to d. Geometrically, this defines how “thin”
        the slab is (w small = more selective).
    angle_steps : int
        Number of discrete directions to test in [0, π). We use [0, π) (not [0, 2π))
        because direction d and -d define the same line, and slabs are symmetric.
        Discretizing angles is a deliberate choice: it bounds runtime while still
        probing many orientations.
    max_tries : int
        Number of random pivot choices. Each try scans all angles. Increasing this
        boosts success probability but increases runtime.

    Returns
    -------
    state : dict or None
        If found, returns a dictionary with the witness configuration:
            'p'       : pivot point (np.ndarray shape (2,))
            'p_index' : index of p within the *ordered list* (not original points array)
            'theta'   : angle of direction d
            'd'       : unit direction vector
            'n'       : unit normal vector (perpendicular to d)
            'R1'      : list of candidate points in backward slab
            'R2'      : list of candidate points in forward slab
            'q1'      : selected point in R1
            'q2'      : selected point in R2
        If no configuration is found within the budget, returns None.

    Design choices (why it is implemented this way)
    -----------------------------------------------
    - Random pivot selection:
        Backtracks may be sparse. Randomly sampling pivots is a cheap way to explore
        many local neighborhoods in the order without O(N^2) enumeration.

    - Discrete angle scan:
        Testing a finite set of directions makes the algorithm predictable and fast.
        In practice, many backtracks are robust to small angle perturbations, so a
        reasonably fine grid (e.g., 180 steps = 1 degree) often works well.

    - Restricting candidates to points *after* p:
        This matches the “later in the linear order” structure used for a witness.

    - Order-consistency constraint pos(q1) < pos(q2):
        We only accept pairs that reflect the intended backtracking pattern in the
        induced path; it prevents trivial geometric hits that do not correspond to
        a real backtrack along the order.

    Notes
    -----
    This is not guaranteed to find a backtrack even if one exists.
    If you need higher recall:
        - increase angle_steps and/or max_tries,
        - or scan more pivots deterministically (more expensive),
        - or adapt (l,w) to the local scale (e.g., dyadic square size).
    """
    import numpy as np
    import random
    from math import cos, sin, pi

    # Ensure numeric array and a concrete ordered list of point coordinates.
    points = np.asarray(points, dtype=float)
    ordered = [points[i] for i in order]

    # A backtrack needs at least 3 points (p, q1, q2).
    if len(ordered) < 3:
        return None

    # Map each *object* (numpy array) to its position in the order.
    # We use id(...) because the elements of `ordered` are numpy arrays,
    # which are not hashable by value.
    # This lets us enforce pos(q1) < pos(q2) quickly.
    pos = {id(q): i for i, q in enumerate(ordered)}

    # Discretize directions:
    # - Use angles in [0, π) because direction d and -d represent the same axis.
    # - endpoint=False avoids duplicating π (same as 0).
    angles = np.linspace(0.0, pi, int(angle_steps), endpoint=False)

    # Try multiple random pivots to increase chance of finding a witness.
    # We avoid picking pivots too close to the ends so "after p" is nontrivial.
    for _ in range(int(max_tries)):

        if len(ordered) < 6:
            # If the ordered set is tiny, the safety margins "2..-3" can break.
            # Fall back to a safer pivot range.
            p_index = random.randint(0, len(ordered) - 1)
        else:
            p_index = random.randint(2, len(ordered) - 3)

        p = ordered[p_index]

        # For each pivot, scan a grid of directions.
        for theta in angles:
            # Unit direction vector d and its perpendicular unit normal n.
            d = np.array([cos(theta), sin(theta)], dtype=float)
            d_norm = np.linalg.norm(d)
            if d_norm == 0:
                continue
            d /= d_norm

            # Rotate d by +90 degrees to get normal direction.
            n = np.array([-d[1], d[0]], dtype=float)

            # Candidate sets in backward / forward slabs.
            R1, R2 = [], []

            # Only consider points that appear after the pivot in the linear order.
            # This matches the witness definition along the induced path.
            for q in ordered[p_index + 1:]:
                v = q - p

                # Projection along the direction d (length-wise coordinate).
                tproj = float(np.dot(v, d))

                # Projection along the normal n (width-wise coordinate).
                uproj = float(np.dot(v, n))

                # Enforce slab thickness: keep only points close to the axis line.
                if abs(uproj) > (w / 2.0):
                    continue

                # Backward slab: behind the pivot along d, within length l.
                if (-l) <= tproj < 0.0:
                    R1.append(q)

                # Forward slab: ahead of the pivot along d, within length l.
                elif 0.0 < tproj <= l:
                    R2.append(q)

            # Build valid (q1, q2) pairs:
            # - q1 in backward slab, q2 in forward slab,
            # - and q1 must appear earlier than q2 in the order.
            valid_pairs = [
                (r1, r2)
                for r1 in R1
                for r2 in R2
                if pos[id(r1)] < pos[id(r2)]
            ]

            # If we have at least one valid witness, pick one uniformly at random.
            # Random choice avoids bias toward any specific geometric pattern.
            if valid_pairs:
                q1, q2 = random.choice(valid_pairs)
                return {
                    "p": p,
                    "p_index": p_index,
                    "theta": float(theta),
                    "d": d,
                    "n": n,
                    "R1": R1,
                    "R2": R2,
                    "q1": q1,
                    "q2": q2,
                }

    # No witness found within the pivot+angle budget.
    return None

def scale_lw_with_t(t, alpha=0.6, beta=0.2):
    s = 2 ** (-t)
    return alpha * s, beta * s

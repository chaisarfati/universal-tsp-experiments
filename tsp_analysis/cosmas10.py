import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.widgets import Button
import math

# === dépendances projet ===
from .geometry import (
    generate_grid_points,
    generate_dyadic_squares,
    filter_points_in_square,
    find_backtrack_state,
)
from .heuristics import platzman_order, zcurve_order, hilbert_order
from .tsp_solver import compute_path_cost, solve_tsp_with_lkh




# ============================================================
# GEOMETRIE DROITE GLOBALE
# ============================================================
def generate_random_global_line():
    theta = np.random.uniform(0, math.pi)
    d = np.array([math.cos(theta), math.sin(theta)])
    n = np.array([-d[1], d[0]])
    p0 = np.random.uniform(0, 1, size=2)
    return p0, d, n, theta


def point_distance_to_line(x, p0, n):
    return abs(np.dot(n, x - p0))


def line_square_intersections(p0, d):
    ts = []
    x0, y0 = p0
    dx, dy = d

    for x in [0, 1]:
        if abs(dx) > 1e-12:
            t = (x - x0) / dx
            y = y0 + t * dy
            if 0 <= y <= 1:
                ts.append(t)

    for y in [0, 1]:
        if abs(dy) > 1e-12:
            t = (y - y0) / dy
            x = x0 + t * dx
            if 0 <= x <= 1:
                ts.append(t)

    if len(ts) < 2:
        return None

    t1, t2 = min(ts), max(ts)
    A = p0 + t1 * d
    B = p0 + t2 * d
    return A, B


def build_strip_polygon(p0, d, n, delta):
    seg1 = line_square_intersections(p0 + delta * n, d)
    seg2 = line_square_intersections(p0 - delta * n, d)
    if seg1 is None or seg2 is None:
        return None
    A1, B1 = seg1
    A2, B2 = seg2
    return Polygon([A1, B1, B2, A2], closed=True)


# ============================================================
# ORDRE RESTREINT
# ============================================================
def restrict_order_to_square(order, square_indices_set):
    return [i for i in order if i in square_indices_set]


# ============================================================
# COLLECTE DES BACKTRACKS (PIVOTS)
# ============================================================
def scale_lw_with_t(M, t, alpha=0.6, beta=0.2):
    s = 2 ** (-t)
    l = alpha * s
    w = beta * s
    return l, w


def collect_pivots_multiscale(points, order, r):
    results = {}
    for t in range(r + 1):
        squares = generate_dyadic_squares(t)
        l, w = scale_lw_with_t(len(points), t)

        for sq_idx, ((x0, y0), size) in enumerate(squares):
            mask = filter_points_in_square(points, x0, y0, size)
            idx = np.where(mask)[0]
            if len(idx) < 3:
                continue

            square_order = restrict_order_to_square(order, set(idx))
            state = find_backtrack_state(
                points, square_order, l=l, w=w,
                angle_steps=180, max_tries=80
            )
            if state is None:
                continue

            state["_square"] = {"t": t, "sq_idx": sq_idx}
            results[(t, sq_idx)] = state
    return results


# ============================================================
# EXTRACTION DE S = PIVOTS DANS LE STRIP
# ============================================================
def extract_strip_pivots(results, global_line, delta):
    p0, d, n, _ = global_line
    S = []
    for st in results.values():
        p = np.array(st["p"])
        if point_distance_to_line(p, p0, n) <= delta:
            S.append(tuple(p))
    return np.array(sorted(set(S)), dtype=float)


def induced_order(global_points, global_order, subset_points):
    index_map = {tuple(p): i for i, p in enumerate(subset_points)}
    return np.array(
        [index_map[tuple(global_points[i])]
         for i in global_order
         if tuple(global_points[i]) in index_map],
        dtype=int
    )


# ============================================================
# VISUALISATION ORDRE vs TSP
# ============================================================
def display_order_vs_tsp(points, order_path, tsp_path,
                         heuristic_cost, tsp_cost):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Order path vs Exact TSP")

    ax.scatter(points[:, 0], points[:, 1], s=30, color="black")

    order_line, = ax.plot(
        points[order_path, 0],
        points[order_path, 1],
        linewidth=2, label="Order path"
    )

    tsp_line, = ax.plot(
        points[tsp_path, 0],
        points[tsp_path, 1],
        linewidth=2, label="Exact TSP"
    )

    ax.legend()

    class Toggle:
        def __init__(self, line):
            self.line = line
            self.visible = True

        def toggle(self, event):
            self.visible = not self.visible
            self.line.set_visible(self.visible)
            fig.canvas.draw_idle()

    btn1 = Button(plt.axes([0.1, 0.02, 0.35, 0.06]), "Show / Hide Order Path")
    btn2 = Button(plt.axes([0.55, 0.02, 0.35, 0.06]), "Show / Hide TSP Path")

    btn1.on_clicked(Toggle(order_line).toggle)
    btn2.on_clicked(Toggle(tsp_line).toggle)

    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # -------------------------
    # 1) POINTS ET ORDRE
    # -------------------------
    M = 512
    points = generate_grid_points(M)
    order, _ = hilbert_order(points)

    # -------------------------
    # 2) BACKTRACKS MULTI-t
    # -------------------------
    r = 5
    results = collect_pivots_multiscale(points, order, r)

    print(f"Total pivots found: {len(results)}")

    # -------------------------
    # 3) DROITE GLOBALE
    # -------------------------
    global_line = generate_random_global_line()
    delta = 0.03

    # -------------------------
    # 4) ENSEMBLE S
    # -------------------------
    S = extract_strip_pivots(results, global_line, delta)
    print(f"|S| = {len(S)}")

    if len(S) < 3:
        print("Not enough points in S.")
        exit()

    # -------------------------
    # 5) CHEMINS
    # -------------------------
    heuristic_indices = induced_order(points, order, S)
    heuristic_cost = compute_path_cost(S, heuristic_indices)

    tsp_indices = solve_tsp_with_lkh(S)
    tsp_cost = compute_path_cost(S, np.array(tsp_indices))

    ratio = heuristic_cost / tsp_cost

    print("\n=== RESULTS ON S ===")
    print(f"Order path cost : {heuristic_cost:.6f}")
    print(f"TSP cost        : {tsp_cost:.6f}")
    print(f"Ratio           : {ratio:.6f}")

    # -------------------------
    # 6) VISUALISATION
    # -------------------------
    display_order_vs_tsp(
        S,
        heuristic_indices,
        np.array(tsp_indices),
        heuristic_cost,
        tsp_cost
    )

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
from .heuristics import hilbert_order
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
    return alpha * s, beta * s


def collect_pivots_multiscale(points, order, r, verbose=True):
    results = {}
    for t in range(r + 1):
        squares = generate_dyadic_squares(t)
        l, w = scale_lw_with_t(len(points), t)

        for sq_idx, ((x0, y0), size) in enumerate(squares):
            mask = filter_points_in_square(points, x0, y0, size)
            idx = np.where(mask)[0].tolist()
            if len(idx) < 3:
                continue

            square_order = restrict_order_to_square(order, set(idx))
            state = find_backtrack_state(
                points, square_order,
                l=l, w=w,
                angle_steps=180,
                max_tries=80
            )
            if state is None:
                continue

            if verbose:
                print(f"[t={t}] square #{sq_idx}: backtrack found")


            state["_square"] = {"t": t, "sq_idx": sq_idx}
            state["_geom"] = {"l": l, "w": w}
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
# VISUALISATION LOCALE : DYADIC SQUARE + BACKTRACK
# ============================================================
def visualize_state_in_square(points, state):
    points = np.asarray(points, float)

    sq = state["_square"]
    t = sq["t"]
    sq_idx = sq["sq_idx"]

    squares = generate_dyadic_squares(t)
    (x0, y0), size = squares[sq_idx]

    mask = filter_points_in_square(points, x0, y0, size)
    idx = np.where(mask)[0].tolist()
    square_order = restrict_order_to_square(list(range(len(points))), set(idx))
    ordered = points[square_order]

    p = state["p"]
    p_index = state["p_index"]
    q1 = state["q1"]
    q2 = state["q2"]
    d = state["d"]
    n = state["n"]
    theta = state["theta"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(x0, x0 + size)
    ax.set_ylim(y0, y0 + size)
    ax.set_title(f"Dyadic square #{sq_idx} (t={t})")
    ax.grid(False)

    # carré dyadique
    ax.add_patch(Rectangle((x0, y0), size, size,
                           edgecolor="gray", facecolor="none"))

    # droite locale L0
    ts = np.linspace(-1, 1, 400)
    line = np.array([p + tt * d for tt in ts])
    ax.plot(line[:, 0], line[:, 1], "--k", label="local line $L_0$")

    # rectangles R1, R2
    l = state["_geom"]["l"]
    w = state["_geom"]["w"]
    # l = np.linalg.norm(q1 - p)
    # w = 0.3 * size
    for sign, color in [(-1, "orange"), (1, "green")]:
        center = p + sign * (l / 2) * d
        corner = center - (l / 2) * d - (w / 2) * n
        ax.add_patch(Rectangle(
            corner, l, w,
            angle=np.degrees(theta),
            facecolor=color,
            edgecolor=color,
            alpha=0.3
        ))

    # points
    before = np.array(ordered[:p_index])
    after = np.array(ordered[p_index + 1:])

    if len(before):
        ax.scatter(before[:, 0], before[:, 1],
                   s=25, color="black", label="before p")
    if len(after):
        ax.scatter(after[:, 0], after[:, 1],
                   s=25, color="cyan", label="after p")

    ax.scatter([p[0]], [p[1]], s=90, color="red", label="pivot p")
    ax.plot([p[0], q1[0], q2[0]],
            [p[1], q1[1], q2[1]],
            color="cyan", linewidth=3, label="backtrack")

    if len(ordered) >= 2:
        ax.plot(ordered[:, 0], ordered[:, 1],
                color="blue", linewidth=1, alpha=0.6,
                label="induced order")

    ax.legend()
    plt.show(block=False)


# ============================================================
# VISUALISATION GLOBALE : ORDRE vs TSP + CLIC
# ============================================================
def display_order_vs_tsp(points, order_path, tsp_path,
                         heuristic_cost, tsp_cost,
                         pivot_state_map):

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Order path vs Exact TSP")

    artist_to_point = {}

    for p in points:
        sc = ax.scatter([p[0]], [p[1]],
                        s=30, color="black",
                        picker=True, zorder=5)
        artist_to_point[sc] = tuple(p)

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

    Button(plt.axes([0.1, 0.02, 0.35, 0.06]),
           "Show / Hide Order Path").on_clicked(Toggle(order_line).toggle)
    Button(plt.axes([0.55, 0.02, 0.35, 0.06]),
           "Show / Hide TSP Path").on_clicked(Toggle(tsp_line).toggle)

    # CLICK HANDLER
    def on_pick(event):
        artist = event.artist
        if artist not in artist_to_point:
            return

        p = artist_to_point[artist]
        if p not in pivot_state_map:
            print("No backtrack state for this point.")
            return

        print(f"[DEBUG] Opening backtrack for pivot {p}")
        visualize_state_in_square(points, pivot_state_map[p])

    fig.canvas.mpl_connect("pick_event", on_pick)

    plt.show()

def display_global_selected_pivots(points,
                                   order,
                                   tsp_order,
                                   pivot_state_map,
                                   title=None):
    """
    Global interactive viewer (pivot-only version).

    - Displays pivot points only (one per dyadic square).
    - Displays two paths:
        * induced linear order
        * exact TSP path
    - Click on a pivot opens its dyadic square and backtrack view.
    - Two buttons:
        * Show / Hide order path
        * Show / Hide TSP path
    """

    points = np.asarray(points, float)

    # --------------------------------------------------
    # Figure + axis (same layout as display_global_selected_triplets)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    button_bottom = 0.02
    button_height = 0.08
    button_top = button_bottom + button_height

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=button_top + 0.025
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)
    ax.grid(False)

    ax.set_title(title or "Global pivots: order vs TSP")

    # --------------------------------------------------
    # Pivot points (cliquables)
    # --------------------------------------------------
    artist_to_state = {}
    pivot_points = np.array(list(pivot_state_map.keys()))

    for p in pivot_points:
        sc = ax.scatter(
            [p[0]], [p[1]],
            s=30,
            color="black",
            picker=True,
            zorder=4
        )
        artist_to_state[sc] = pivot_state_map[tuple(p)]

    # --------------------------------------------------
    # Induced linear order path
    # --------------------------------------------------
    order_path = points[order]
    order_line, = ax.plot(
        order_path[:, 0],
        order_path[:, 1],
        color="blue",
        linewidth=2.0,
        alpha=0.7,
        label="Induced order path"
    )

    # --------------------------------------------------
    # Exact TSP path
    # --------------------------------------------------
    tsp_path = points[tsp_order]
    tsp_line, = ax.plot(
        tsp_path[:, 0],
        tsp_path[:, 1],
        color="orange",
        linewidth=2.0,
        alpha=0.7,
        label="Exact TSP path"
    )

    ax.legend(loc="upper right")

    # --------------------------------------------------
    # Buttons (same philosophy as before)
    # --------------------------------------------------
    class Toggle:
        def __init__(self, artist):
            self.artist = artist
            self.visible = True

        def toggle(self, event):
            self.visible = not self.visible
            self.artist.set_visible(self.visible)
            fig.canvas.draw_idle()

    ax_btn1 = plt.axes([0.10, button_bottom, 0.35, button_height])
    ax_btn2 = plt.axes([0.55, button_bottom, 0.35, button_height])

    btn1 = Button(ax_btn1, "Show / Hide Order Path")
    btn2 = Button(ax_btn2, "Show / Hide TSP Path")

    btn1.on_clicked(Toggle(order_line).toggle)
    btn2.on_clicked(Toggle(tsp_line).toggle)

    # --------------------------------------------------
    # Click handler: open local backtrack view
    # --------------------------------------------------
    local_debug_fig = {"fig": None}

    def on_pick(event):
        artist = event.artist
        if artist not in artist_to_state:
            return

        state = artist_to_state[artist]

        if local_debug_fig["fig"] is not None:
            plt.close(local_debug_fig["fig"])

        fig_local = visualize_state_in_square(points, state)
        local_debug_fig["fig"] = fig_local

    fig.canvas.mpl_connect("pick_event", on_pick)

    # --------------------------------------------------
    # Clean close
    # --------------------------------------------------
    def on_close(event):
        plt.close("all")

    fig.canvas.mpl_connect("close_event", on_close)

    print("[DEBUG] Interactive pivot viewer ready")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    M = 512
    points = generate_grid_points(M)
    order, _ = hilbert_order(points)

    r = 5
    results = collect_pivots_multiscale(points, order, r)
    print(f"Total pivots found: {len(results)}")

    global_line = generate_random_global_line()
    delta = 0.03

    S = extract_strip_pivots(results, global_line, delta)

    pivot_state_map = {
        tuple(st["p"]): st
        for st in results.values()
        if tuple(st["p"]) in map(tuple, S)
    }

    print(f"|S| = {len(S)}")

    heuristic_indices = induced_order(points, order, S)
    heuristic_cost = compute_path_cost(S, heuristic_indices)

    tsp_indices = solve_tsp_with_lkh(S)
    tsp_cost = compute_path_cost(S, np.array(tsp_indices))

    print("\n=== RESULTS ON S ===")
    print(f"Order path cost : {heuristic_cost:.6f}")
    print(f"TSP cost        : {tsp_cost:.6f}")
    print(f"Ratio           : {heuristic_cost / tsp_cost:.6f}")

    display_global_selected_pivots(
        points=S,
        order=heuristic_indices,
        tsp_order=np.array(tsp_indices),
        pivot_state_map=pivot_state_map,
        title="Global pivots inside strip: order vs TSP"
    )
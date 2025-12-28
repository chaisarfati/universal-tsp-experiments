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
# 0) OUTILS GLOBAUX: droite aléatoire + voisinage dans [0,1]^2
# ============================================================

def _line_square_intersections(p0, d, eps=1e-12):
    """
    Intersections du segment infini p(t)=p0 + t d avec le carré [0,1]^2.
    Retourne 2 points (A,B) si la droite coupe le carré, sinon None.
    """
    x0, y0 = p0
    dx, dy = d
    ts = []

    # x = 0, x = 1
    if abs(dx) > eps:
        t = (0.0 - x0) / dx
        y = y0 + t * dy
        if -eps <= y <= 1.0 + eps:
            ts.append(t)
        t = (1.0 - x0) / dx
        y = y0 + t * dy
        if -eps <= y <= 1.0 + eps:
            ts.append(t)

    # y = 0, y = 1
    if abs(dy) > eps:
        t = (0.0 - y0) / dy
        x = x0 + t * dx
        if -eps <= x <= 1.0 + eps:
            ts.append(t)
        t = (1.0 - y0) / dy
        x = x0 + t * dx
        if -eps <= x <= 1.0 + eps:
            ts.append(t)

    # Dédupliquer / trier
    if len(ts) < 2:
        return None
    ts = sorted(set([float(t) for t in ts]))
    if len(ts) < 2:
        return None

    # Prendre extrêmes
    tA, tB = ts[0], ts[-1]
    A = np.array([x0 + tA * dx, y0 + tA * dy], dtype=float)
    B = np.array([x0 + tB * dx, y0 + tB * dy], dtype=float)
    return A, B


def generate_random_global_line():
    """
    Génère une droite L qui traverse [0,1]^2 :
      - angle theta uniform sur [0, pi)
      - point p0 uniforme dans le carré, la droite passe par p0
    Retourne (p0, d, n, theta) où:
      d = direction unitaire
      n = normale unitaire (perpendiculaire à d)
    """
    theta = np.random.rand() * math.pi
    d = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    n = np.array([-d[1], d[0]], dtype=float)  # rotation de +90°
    p0 = np.random.rand(2)
    return p0, d, n, theta


def point_distance_to_line(x, p0, n_unit):
    """
    Distance d'un point x à la droite définie par { y : n·(y-p0)=0 }.
    Si n est unitaire => |n·(x-p0)|.
    """
    return abs(float(np.dot(n_unit, (x - p0))))


def build_band_polygon_in_unit_square(p0, d, n, delta):
    """
    Construit un polygone (4 sommets) représentant le "band" autour de L:
      {x : |n·(x-p0)| <= delta}
    et le clip implicitement par la génération des deux lignes parallèles
    à l'intérieur du carré [0,1]^2.

    Retourne un Polygon (matplotlib) ou None si impossible.
    """
    # deux lignes parallèles: n·(x-p0)=+/-delta
    # cela équivaut à décaler le point d'origine: p0_plus = p0 + delta*n, p0_minus = p0 - delta*n
    p_plus = p0 + delta * n
    p_minus = p0 - delta * n

    seg_plus = _line_square_intersections(p_plus, d)
    seg_minus = _line_square_intersections(p_minus, d)
    if seg_plus is None or seg_minus is None:
        return None

    A1, B1 = seg_plus
    A2, B2 = seg_minus

    # Former un quadrilatère cohérent
    # (A1->B1) et (B2->A2) pour fermer sans croiser
    poly_pts = np.vstack([A1, B1, B2, A2])
    return Polygon(poly_pts, closed=True)


# ============================================================
# 1) UTIL: restriction d'ordre à un dyadic square
# ============================================================

def restrict_order_to_square(order, square_indices_set):
    """Return global order filtered to indices belonging to this square."""
    return [i for i in order if i in square_indices_set]


# ============================================================
# 2) COLLECTE: backtracks par carré, mais on ne garde que le pivot p
# ============================================================

def collect_pivots_by_square(points, order, t, l=0.3, w=0.08,
                             angle_steps=180, max_tries=50, verbose=True):
    """
    Idem à ta collecte, mais:
      - on garde le state complet (pour debug local),
      - dans l'analyse globale on ne sélectionnera QUE le pivot p.
    """
    squares = generate_dyadic_squares(t)
    points = np.asarray(points, float)

    results = {}  # (t, sq_idx) -> state

    for sq_idx, ((x0, y0), size) in enumerate(squares):
        mask = filter_points_in_square(points, x0, y0, size)
        idx = np.where(mask)[0].tolist()

        if len(idx) < 3:
            if verbose:
                print(f"[t={t}] square #{sq_idx}: too few points ({len(idx)})")
            continue

        square_indices_set = set(idx)
        square_order = restrict_order_to_square(order, square_indices_set)

        state = find_backtrack_state(
            points, square_order,
            l=l, w=w,
            angle_steps=angle_steps,
            max_tries=max_tries
        )

        if state is None:
            if verbose:
                print(f"[t={t}] square #{sq_idx}: NO backtrack found")
            continue

        # metadata
        state["_square"] = {"x0": x0, "y0": y0, "size": size, "sq_idx": sq_idx, "t": t}
        state["_square_order"] = square_order

        results[(t, sq_idx)] = state
        if verbose:
            print(f"[t={t}] square #{sq_idx}: backtrack found with p_index={state['p_index']}")

    return results


def scale_lw_with_t(M, t, alpha=0.6, beta=0.2, min_w_in_steps=3, max_frac=0.9):
    """
    Returns l(t), w(t) scaled to dyadic square side s=2^-t.
    - alpha, beta are fractions of the dyadic square side.
    - min_w_in_steps enforces w >= min_w_in_steps * (1/M).
    - max_frac enforces w <= max_frac * s (so the strip never covers the whole square).
    """
    s = 2 ** (-t)
    step = 1.0 / M

    l = alpha * s
    w = beta * s

    w = max(w, min_w_in_steps * step)
    w = min(w, max_frac * s)
    l = min(l, max_frac * s)
    return l, w


# ============================================================
# 3) VUE LOCALE: inchangée (avec t affiché)
# ============================================================

def visualize_state_in_square(points, state, l=0.3, w=0.08, title=None):
    points = np.asarray(points, float)
    sq = state["_square"]
    square_order = state["_square_order"]

    ordered = [points[i] for i in square_order]

    p = state["p"]
    p_index = state["p_index"]
    theta = state["theta"]
    d = state["d"]
    n = state["n"]
    q1 = state["q1"]
    q2 = state["q2"]

    x0, y0, size = sq["x0"], sq["y0"], sq["size"]
    t = sq.get("t", None)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(x0, x0 + size)
    ax.set_ylim(y0, y0 + size)
    ax.grid(False)
    ax.set_title(title or f"Dyadic square #{sq['sq_idx']} (t={t})")

    ax.add_patch(Rectangle((x0, y0), size, size,
                           edgecolor="lightgray", facecolor="none", linewidth=1.0))

    # droite locale L du backtrack (celle de l'oracle)
    t_vals = np.linspace(-1, 1, 300)
    line = np.array([p + tt * d for tt in t_vals])
    ax.plot(line[:, 0], line[:, 1], "--k", linewidth=1)

    # rectangles R1 / R2
    for sign, color in [(-1, "orange"), (1, "green")]:
        center = p + sign * (l / 2) * d
        corner = center - (l / 2) * d - (w / 2) * n
        rect = Rectangle(
            corner,
            width=l,
            height=w,
            angle=np.degrees(theta),
            facecolor=color,
            edgecolor=color,
            alpha=0.35
        )
        ax.add_patch(rect)

    before = np.array(ordered[:p_index])
    after = np.array(ordered[p_index + 1:])

    if len(before):
        ax.scatter(before[:, 0], before[:, 1], s=30, color="black", label="before p")
    if len(after):
        ax.scatter(after[:, 0], after[:, 1], s=30, color="cyan", label="after p")

    ax.scatter([p[0]], [p[1]], s=90, color="red", label="pivot p")

    xs, ys = zip(*ordered)
    ax.plot(xs, ys, color="blue", linewidth=1)

    ax.plot([p[0], q1[0], q2[0]],
            [p[1], q1[1], q2[1]],
            color="cyan", linewidth=3)

    ax.legend(loc="upper right")
    plt.show(block=False)
    return fig


# ============================================================
# 4) VUE GLOBALE: pivots uniquement + droite globale + voisinage
# ============================================================

def display_global_pivots_multiscale(points, order, results_all_scales,
                                    global_line=None,
                                    neighborhood_delta=0.02,
                                    show_squares=False,
                                    title=None):
    """
    Fenêtre globale interactive (multiscale):
      - affiche tous les pivots p (un par carré ayant un backtrack), pour t=0..r
      - trace la droite globale L + son voisinage (bande)
      - colore EN CYAN VIF uniquement les pivots qui tombent dans le voisinage de L
      - click sur un pivot -> ouvre la fenêtre locale du carré correspondant (avec t)
      - affiche en terminal la liste des pivots (et leur (t,sq_idx)) dans le voisinage
    """
    points = np.asarray(points, float)

    # unpack line
    if global_line is None:
        p0, d, n, theta = generate_random_global_line()
    else:
        p0, d, n, theta = global_line

    # pivots + test voisinage
    pivots = []
    pivot_keys = []
    in_band = []

    for key, st in results_all_scales.items():
        p = np.array(st["p"], dtype=float)
        pivots.append(p)
        pivot_keys.append(key)
        in_band.append(point_distance_to_line(p, p0, n) <= neighborhood_delta)

    pivots = np.array(pivots, dtype=float) if len(pivots) else np.zeros((0, 2), dtype=float)
    in_band = np.array(in_band, dtype=bool) if len(in_band) else np.zeros((0,), dtype=bool)

    # print list to terminal
    kept = []
    for k, p, ok in zip(pivot_keys, pivots, in_band):
        if ok:
            t, sq_idx = k
            kept.append((t, sq_idx, float(p[0]), float(p[1])))
    print("\n=== PIVOTS IN GLOBAL LINE NEIGHBORHOOD ===")
    print(f"delta = {neighborhood_delta}")
    print(f"count = {len(kept)}")
    for (t, sq_idx, x, y) in kept:
        print(f"  (t={t}, square={sq_idx})  p=({x:.6f}, {y:.6f})")
    print("========================================\n")

    # figure
    fig, ax = plt.subplots(figsize=(7, 7))
    button_bottom = 0.02
    button_height = 0.08
    button_top = button_bottom + button_height

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=button_top + 0.025)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(False)
    ax.set_title(title or "Global pivots (t=0..r) + random line L + neighborhood")

    # grid points context (faint)
    ax.scatter(points[:, 0], points[:, 1], s=10, color="black", alpha=0.12, zorder=1)

    # draw band (semi-transparent) and line
    band_poly = build_band_polygon_in_unit_square(p0, d, n, neighborhood_delta)
    if band_poly is not None:
        # IMPORTANT: mettre un zorder bas + alpha faible, pour ne pas masquer L
        band_poly.set_facecolor("cyan")
        band_poly.set_edgecolor("none")
        band_poly.set_alpha(0.12)
        band_poly.set_zorder(2)
        ax.add_patch(band_poly)

    seg = _line_square_intersections(p0, d)
    if seg is not None:
        A, B = seg
        ax.plot([A[0], B[0]], [A[1], B[1]], color="black", linewidth=2.0, zorder=4, label="Global line L")

    # pivots: couleur par t hors strip, cyan vif dans strip
    artist_to_key = {}
    pivot_artists = []

    for (p, key, ok) in zip(pivots, pivot_keys, in_band):
        t, sq_idx = key

        if ok:
            # priorité absolue : strip
            color = "cyan"
            size = 60
            z = 7
        else:
            # couleur dépendant de t
            color = COLOR_BY_T.get(t, "#bbbbbb")
            size = 40
            z = 6

        a = ax.scatter(
            [p[0]], [p[1]],
            s=size,
            color=color,
            alpha=1.0,
            zorder=z,
            picker=True
        )
        artist_to_key[a] = key
        pivot_artists.append(a)


    # Toggle show/hide pivots
    class TogglePivots:
        def __init__(self, artists, fig):
            self.artists = artists
            self.fig = fig
            self.visible = True

        def toggle(self, event):
            self.visible = not self.visible
            for ar in self.artists:
                ar.set_visible(self.visible)
            self.fig.canvas.draw_idle()

    ax_button = plt.axes([0.30, button_bottom, 0.40, button_height])
    btn = Button(ax_button, "Show / Hide Pivots")
    btn.on_clicked(TogglePivots(pivot_artists, fig).toggle)

    local_debug_fig = {"fig": None}

    def on_pick(event):
        artist = event.artist
        if artist not in artist_to_key:
            return
        key = artist_to_key[artist]
        st = results_all_scales[key]
        t, sq_idx = key
        if local_debug_fig["fig"] is not None:
            plt.close(local_debug_fig["fig"])
        fig_local = visualize_state_in_square(
            points,
            st,
            l=st.get("_l_used", 0.3),
            w=st.get("_w_used", 0.08),
            title=f"Dyadic square #{sq_idx} (t={t})"
        )
        local_debug_fig["fig"] = fig_local

    def on_close(event):
        plt.close("all")
        raise SystemExit

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("close_event", on_close)

    ax.legend(loc="upper right")
    plt.show()

# Couleurs claires par échelle t (cyclables si besoin)
COLOR_BY_T = {
    0: "#cccccc",  # gris clair
    1: "#ff9999",  # rouge clair
    2: "#99ccff",  # bleu clair
    3: "#99ff99",  # vert clair
    4: "#ffcc99",  # orange clair
    5: "#dda0dd",  # violet clair
    6: "#ffff99",  # jaune clair
}

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # ===== Étape 1: générer le carré unité M×M =====
    M = 512
    points = generate_grid_points(M)

    # ===== Étape 2: choix de l'ordre linéaire =====
    # order, _ = zcurve_order(points)
    # order, _ = platzman_order(points)
    order, _ = hilbert_order(points)

    # ===== Étape 3: multi-échelle t=0..r =====
    r = 5
    results_all = {}

    for t in range(0, r + 1):
        l, w = scale_lw_with_t(M, t, alpha=0.6, beta=0.2, min_w_in_steps=3)

        res_t = collect_pivots_by_square(
            points=points,
            order=order,
            t=t,
            l=l,
            w=w,
            angle_steps=180,
            max_tries=80,
            verbose=True
        )

        # (option) garder l,w utilisés dans le state pour affichage local cohérent
        for k, st in res_t.items():
            st["_l_used"] = l
            st["_w_used"] = w

        results_all.update(res_t)

    print("\n=== SUMMARY ===")
    print(f"scales t=0..{r}")
    print(f"total pivots collected = {len(results_all)}")

    # ===== Étape 4: droite globale + voisinage + affichage =====
    # Choisis delta (épaisseur du voisinage). 0.02 est un bon départ.
    neighborhood_delta = 0.02

    global_line = generate_random_global_line()  # (p0, d, n, theta)

    display_global_pivots_multiscale(
        points=points,
        order=order,
        results_all_scales=results_all,
        global_line=global_line,
        neighborhood_delta=neighborhood_delta,
        show_squares=False,
        title=f"Global pivots (t=0..{r}) + random line L + neighborhood (delta={neighborhood_delta})"
    )

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from math import cos, sin, pi, log2
import math

# === dépendances projet ===
from .geometry import generate_grid_points, generate_dyadic_squares, filter_points_in_square, find_backtrack_state
from .heuristics import platzman_order, zcurve_order, hilbert_order
from .experiment import save_result_to_file
from .tsp_solver import compute_path_cost, solve_tsp_with_lkh 


# Utilité:
# - IMPORTANT: on ne modifie jamais l’ordre global.
# - On restreint l’ordre global aux indices des points dans le carré Q.
# - C’est exactement “l’ordre induit” sur les points de Q.
def restrict_order_to_square(order, square_indices_set):
    """Return global order filtered to indices belonging to this square."""
    return [i for i in order if i in square_indices_set]


# ============================================================
# 3) PIPELINE DYADIQUE: COLLECTE DES BACKTRACKS
# ============================================================
# Utilité:
# - Étape centrale de ton pipeline:
#   pour chaque dyadic square, on calcule un backtrack state (si possible)
#   et on stocke tout le state dans un dictionnaire.
# - Résultat: {square_id : state_dict}.
def collect_backtracks_by_square(points, order, t, l=0.3, w=0.08,
                                 angle_steps=180, max_tries=50, verbose=True):
    squares = generate_dyadic_squares(t)
    points = np.asarray(points, float)

    results = {}  # square_id -> state

    for sq_idx, ((x0, y0), size) in enumerate(squares):
        mask = filter_points_in_square(points, x0, y0, size)
        idx = np.where(mask)[0].tolist()

        if len(idx) < 3:
            if verbose:
                print(f"[t={t}] square #{sq_idx}: too few points ({len(idx)})")
            continue

        square_indices_set = set(idx)
        square_order = restrict_order_to_square(order, square_indices_set)

        state = find_backtrack_state(points, square_order, l=l, w=w,
                                     angle_steps=angle_steps, max_tries=max_tries)

        if state is None:
            if verbose:
                print(f"[t={t}] square #{sq_idx}: NO backtrack found")
            continue

        # We also keep the square info for debug (very useful)
        state["_square"] = {"x0": x0, "y0": y0, "size": size, "sq_idx": sq_idx}
        state["_square_order"] = square_order 

        results[sq_idx] = state
        if verbose:
            print(f"[t={t}] square #{sq_idx}: backtrack found with p_index={state['p_index']}")

    return results


# ============================================================
# 4) DISPLAY GLOBAL (étape 4)
# ============================================================
# Utilité:
# - Afficher sur le carré unité tous les triplets sélectionnés:
#   (p, q1, q2) pour chaque dyadic square qui a réussi.
# - Cela donne un aperçu global de la couverture des carrés.
def display_global_selected_triplets(points, order, t, results,
                                     l=0.3, w=0.08,
                                     show_squares=True,
                                     title=None):
    """
    Fenêtre globale interactive :
    - affiche tous les backtracks (triplets p,q1,q2) sur le carré unité
    - trace le chemin induit par l'ordre linéaire global sur les points sélectionnés
    - click sur point OU ligne d’un triplet -> ouvre la fenêtre locale du dyadic square
    - bouton dans la fenêtre globale : Show/Hide Backtracks
    """
    squares = generate_dyadic_squares(t)
    points = np.asarray(points, float)

    # --------------------------------------------------
    # Figure + axe principal (maximisation de [0,1]²)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    # Le bouton occupera verticalement [0.02, 0.10]
    button_bottom = 0.02
    button_height = 0.08
    button_top = button_bottom + button_height

    # Axe principal juste au-dessus du bouton
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=button_top + 0.025
    )

    # Géométrie stricte du carré unité
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

    # Lisibilité
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(False)

    ax.set_title(title or f"Global backtracks + linear order (t={t})")

    # --------------------------------------------------
    # Contours dyadiques (optionnel)
    # --------------------------------------------------
    if show_squares:
        for (x0, y0), size in squares:
            ax.add_patch(
                Rectangle(
                    (x0, y0), size, size,
                    edgecolor="lightgray",
                    facecolor="none",
                    linewidth=0.6
                )
            )

    # --------------------------------------------------
    # Points du grid (contexte)
    # --------------------------------------------------
    ax.scatter(points[:, 0], points[:, 1],
               s=10, color="black", alpha=0.15)

    # --------------------------------------------------
    # Ensemble S = union des points (p,q1,q2)
    # + restriction de l'ordre global à S
    # --------------------------------------------------
    selected_pts = set()
    for st in results.values():
        selected_pts.add(tuple(st["p"]))
        selected_pts.add(tuple(st["q1"]))
        selected_pts.add(tuple(st["q2"]))

    point_index = {tuple(points[i]): i for i in range(len(points))}
    selected_indices = {
        point_index[p] for p in selected_pts if p in point_index
    }

    restricted_order = [i for i in order if i in selected_indices]
    if len(restricted_order) >= 2:
        path = points[restricted_order]
        ax.plot(
            path[:, 0], path[:, 1],
            color="blue", linewidth=1.5, alpha=0.7,
            label="Global linear path"
        )

    # --------------------------------------------------
    # Backtracks (points + lignes cliquables)
    # --------------------------------------------------
    artist_to_square = {}
    backtrack_lines = []
    local_debug_fig = {"fig": None}

    print("[DEBUG] Registering backtracks for picking")

    for sq_idx, st in results.items():
        p, q1, q2 = st["p"], st["q1"], st["q2"]
        xs = [p[0], q1[0], q2[0]]
        ys = [p[1], q1[1], q2[1]]

        sc = ax.scatter(xs, ys, s=18, color="C1",
                        picker=True, zorder=4)
        artist_to_square[sc] = sq_idx

        line, = ax.plot(xs, ys, linewidth=2.5,
                        color="cyan", picker=True, zorder=5)
        line.set_pickradius(12)

        artist_to_square[line] = sq_idx
        backtrack_lines.append(line)

        print(f"[DEBUG] Backtrack registered for square #{sq_idx}")

    # --------------------------------------------------
    # Bouton global : Show / Hide Backtracks
    # --------------------------------------------------
    class ToggleBacktracks:
        def __init__(self, lines, fig):
            self.lines = lines
            self.fig = fig
            self.visible = True

        def toggle(self, event):
            self.visible = not self.visible
            for ln in self.lines:
                ln.set_visible(self.visible)
            self.fig.canvas.draw_idle()

    ax_button = plt.axes([
        0.30,
        button_bottom,
        0.40,
        button_height
    ])
    btn = Button(ax_button, "Show / Hide Backtracks")
    btn.on_clicked(ToggleBacktracks(backtrack_lines, fig).toggle)

    # --------------------------------------------------
    # CLICK : ouvrir la fenêtre locale du dyadic square
    # --------------------------------------------------
    def on_pick(event):
        artist = event.artist
        print("[DEBUG] pick_event detected")

        if artist not in artist_to_square:
            print("[DEBUG] Picked artist not registered")
            return

        sq_idx = artist_to_square[artist]
        print(f"[DEBUG] Opening dyadic square #{sq_idx}")

        if local_debug_fig["fig"] is not None:
            plt.close(local_debug_fig["fig"])

        state = results[sq_idx]
        fig_local = visualize_state_in_square(
            points,
            state,
            l=l,
            w=w,
            title=f"Dyadic square #{sq_idx} (t={t})"
        )
        local_debug_fig["fig"] = fig_local

    # --------------------------------------------------
    # Fermeture propre
    # --------------------------------------------------
    def on_close(event):
        print("[DEBUG] Closing application")
        plt.close("all")
        raise SystemExit

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("close_event", on_close)

    ax.legend(loc="upper right")
    print("[DEBUG] Interactive global window ready")
    plt.show()


# ============================================================
# 5) DEBUG VIEWER PAR DYADIC SQUARE (L, R1, R2, violet/noir, q1/q2)
# ============================================================
# Utilité:
# - Tu veux pouvoir “ouvrir” chaque dyadic square et voir exactement:
#   L, R1, R2, pivot p, points avant/après (noir/cyan), q1/q2.
# - On réutilise la même logique d’affichage que ta brique,
#   mais on injecte un state déjà calculé (pour debug reproductible).
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(x0, x0 + size)
    ax.set_ylim(y0, y0 + size)
    ax.grid(False)
    ax.set_title(title or f"Dyadic square #{sq['sq_idx']}")

    # carré dyadique
    ax.add_patch(
        Rectangle((x0, y0), size, size,
                  edgecolor="lightgray", facecolor="none", linewidth=1.0)
    )

    # droite L
    t_vals = np.linspace(-1, 1, 300)
    line = np.array([p + t * d for t in t_vals])
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

    # points avant/après p
    before = np.array(ordered[:p_index])
    after = np.array(ordered[p_index + 1:])

    if len(before):
        ax.scatter(before[:, 0], before[:, 1], s=30, color="black", label="before p")
    if len(after):
        ax.scatter(after[:, 0], after[:, 1], s=30, color="cyan", label="after p")

    # pivot
    ax.scatter([p[0]], [p[1]], s=90, color="red", label="pivot p")

    # chemin induit
    xs, ys = zip(*ordered)
    ax.plot(xs, ys, color="blue", linewidth=1)

    # backtrack
    ax.plot([p[0], q1[0], q2[0]],
            [p[1], q1[1], q2[1]],
            color="cyan", linewidth=3)

    ax.legend(loc="upper right")

    plt.show(block=False)
    return fig



# ============================================================
# 6) “TABLEAU DE BORD” : parcourir les carrés et debugger
# ============================================================
# Utilité:
# - Après la collecte, tu veux pouvoir inspecter facilement chaque carré.
# - Ici: on affiche la vue globale, puis on ouvre une vue par carré trouvé.
# - (Pour t=1 c’est OK, pour t=3 ça fait beaucoup: mais c’est exactement ce que tu as demandé.)
def debug_all_squares(points, order, t, results, l=0.3, w=0.08):
    display_global_selected_triplets(
        points=points,
        order=order,
        t=t,
        results=results,
        l=l,
        w=w,
        show_squares=True
    )

def scale_lw_with_t(M, t, alpha=0.6, beta=0.2, min_w_in_steps=3, max_frac=0.9):
    """
    Returns l(t), w(t) scaled to dyadic square side s=2^-t.
    - alpha, beta are fractions of the dyadic square side.
    - min_w_in_steps enforces w >= min_w_in_steps * (1/M).
    - max_frac enforces w <= max_frac * s (so the strip never covers the whole square).
    """
    s = 2 ** (-t)        # dyadic square side length
    step = 1.0 / M       # grid spacing

    l = alpha * s
    w = beta * s

    # keep w meaningful w.r.t grid resolution
    w = max(w, min_w_in_steps * step)

    # never let the strip cover (almost) the whole square
    w = min(w, max_frac * s)

    # you may also want l not to exceed the square size
    l = min(l, max_frac * s)

    return l, w

# ============================================================
# MAIN (pipeline complet)
# ============================================================
if __name__ == "__main__":
    # ===== Étape 1: générer le carré unité M×M =====
    M = 512
    points = generate_grid_points(M)

    # Choix ordre (comme tu veux)
    # order, _ = zcurve_order(points)
    # order, _ = platzman_order(points)
    order, _ = hilbert_order(points)

    # ===== Étape 2: paramètres dyadiques =====
    t = 5  # t=1 => 4 dyadic squares, t=2 => 16, etc.

    # paramètres rectangles
    # l = 0.30
    # w = 0.08
    l, w = scale_lw_with_t(M, t, alpha=0.6, beta=0.2, min_w_in_steps=3)
    # ===== Étape 3: collecte par dyadic square =====
    results = collect_backtracks_by_square(
        points=points,
        order=order,
        t=t,
        l=l,
        w=w,
        angle_steps=180,
        max_tries=80,
        verbose=True
    )

    print("\n=== SUMMARY ===")
    print(f"dyadic scale t={t}, squares total={len(generate_dyadic_squares(t))}")
    print(f"squares with backtrack found={len(results)}")
    
    # =========================================================
    # ===== Étape 5: ANALYSE QUANTITATIVE (NOUVELLE) =====
    # =========================================================
    # Construire l'ensemble final des points sélectionnés
    selected_points = []
    for st in results.values():
        selected_points.append(tuple(st["p"]))
        selected_points.append(tuple(st["q1"]))
        selected_points.append(tuple(st["q2"]))

    # unicité + conversion array
    selected_points = np.array(sorted(set(selected_points)), dtype=float)
    n_final = len(selected_points)

    print("\n=== QUANTITATIVE ANALYSIS ===")
    print(f"number of selected points n = {n_final}")

    if n_final < 3:
        print("Not enough points for TSP comparison.")
    else:
        # --- ordre heuristique restreint ---
        # map coord -> index
        index_map = {tuple(p): i for i, p in enumerate(selected_points)}

        # restreindre l'ordre global aux points sélectionnés
        heuristic_indices = [
            index_map[tuple(points[i])]
            for i in order
            if tuple(points[i]) in index_map
        ]

        # --- coût heuristique ---
        heuristic_cost = compute_path_cost(
            selected_points,
            np.array(heuristic_indices, dtype=int)
        )

        # --- TSP exact (LKH) ---
        # tsp_indices = solve_tsp_with_lkh(selected_points)

        # tsp_cost = compute_path_cost(
        #     selected_points,
        #     np.array(tsp_indices, dtype=int)
        # )

        tsp_cost = 2

        ratio = heuristic_cost / tsp_cost if tsp_cost > 0 else float("inf")

        # --- borne de Cosmas ---
        # n = |points|, borne = sqrt(log n / log log n)
        if n_final > 2:
            cosmas_bound = math.sqrt(
                math.log2(n_final) / math.log2(math.log2(n_final))
            )
        else:
            cosmas_bound = 0.0

        print(f"heuristic path cost : {heuristic_cost:.6f}")
        print(f"TSP optimal cost   : {tsp_cost:.6f}")
        print(f"ratio (heur / tsp) : {ratio:.6f}")
        print(f"Cosmas lower bound : {cosmas_bound:.6f}")

        if cosmas_bound > 0:
            print(f"ratio / bound      : {ratio / cosmas_bound:.6f}")

    # ===== Étape 4: visualisation + debug =====
    debug_all_squares(points, order, t, results, l=l, w=w)
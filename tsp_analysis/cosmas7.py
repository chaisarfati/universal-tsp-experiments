import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from math import cos, sin, pi, log2
import random

# === dépendances projet (comme chez toi) ===
from .geometry import generate_grid_points
from .heuristics import platzman_order, zcurve_order, hilbert_order
from .experiment import save_result_to_file


# ============================================================
# 1) DYADIC SQUARES
# ============================================================
# Utilité:
# - Le papier partitionne [0,1]^2 en carrés dyadiques de taille 1/2^t.
# - On doit itérer sur chacun d'eux et y chercher un backtrack local.
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


# Utilité:
# - Pour un dyadic square Q, on doit filtrer les points qui appartiennent à Q.
# - On utilise l’intervalle [x0,x1) et [y0,y1) pour éviter les doublons sur les frontières.
def filter_points_in_square(points, x0, y0, size):
    """Return mask selecting points inside dyadic square [x0,x1)×[y0,y1)."""
    x1, y1 = x0 + size, y0 + size
    pts = np.asarray(points, float)
    mask = (
        (pts[:, 0] >= x0) & (pts[:, 0] < x1) &
        (pts[:, 1] >= y0) & (pts[:, 1] < y1)
    )
    return mask


# Utilité:
# - IMPORTANT: on ne modifie jamais l’ordre global.
# - On restreint l’ordre global aux indices des points dans le carré Q.
# - C’est exactement “l’ordre induit” sur les points de Q.
def restrict_order_to_square(order, square_indices_set):
    """Return global order filtered to indices belonging to this square."""
    return [i for i in order if i in square_indices_set]


# ============================================================
# 2) BACKTRACK ORACLE (TA SIGNATURE)
# ============================================================
# Utilité:
# - C’est ta brique de confiance: pivot aléatoire, balayage des directions,
#   et définition directionnelle via L, R1, R2 avec paramètres l,w.
# - On la conserve exactement et on l’utilise comme “oracle”.
def find_backtrack_state(points, order, l=0.3, w=0.08, angle_steps=180, max_tries=50):
    points = np.array(points, float)
    ordered = [points[i] for i in order]

    if len(ordered) < 3:
        return None

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

            if R1 and R2:
                return {
                    "p": p,
                    "p_index": p_index,
                    "theta": theta,
                    "d": d,
                    "n": n,
                    "R1": R1,
                    "R2": R2,
                    "q1": random.choice(R1),
                    "q2": random.choice(R2),
                }

    return None


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

        # Appel de ta brique de confiance (oracle)
        state = find_backtrack_state(points, square_order, l=l, w=w,
                                     angle_steps=angle_steps, max_tries=max_tries)

        if state is None:
            if verbose:
                print(f"[t={t}] square #{sq_idx}: NO backtrack found")
            continue

        # On garde aussi info sur le carré pour debug (très utile)
        state["_square"] = {"x0": x0, "y0": y0, "size": size, "sq_idx": sq_idx}
        state["_square_order"] = square_order  # ordre induit dans ce carré

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
def display_global_selected_triplets(points, t, results, show_squares=True, title=None):
    squares = generate_dyadic_squares(t)
    points = np.asarray(points, float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.set_title(title or f"Global selected backtracks — dyadic scale t={t}")

    # Option: afficher les contours dyadiques
    if show_squares:
        for (x0, y0), size in squares:
            ax.add_patch(
                Rectangle((x0, y0), size, size, edgecolor="lightgray", facecolor="none", linewidth=0.6)
            )

    # tous les points (optionnel mais utile comme contexte)
    ax.scatter(points[:, 0], points[:, 1], s=10, color="black", alpha=0.25, label="all grid points")

    # triplets sélectionnés
    for sq_idx, st in results.items():
        p = st["p"]
        q1 = st["q1"]
        q2 = st["q2"]

        xs = [p[0], q1[0], q2[0]]
        ys = [p[1], q1[1], q2[1]]

        ax.scatter(xs, ys, s=50)
        ax.plot(xs, ys, linewidth=2)

    ax.legend(loc="upper right")


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
    plt.subplots_adjust(bottom=0.10)

    ax.set_aspect("equal")
    ax.set_xlim(x0, x0 + size)
    ax.set_ylim(y0, y0 + size)
    ax.grid(False)
    ax.set_title(title or f"Dyadic square #{sq['sq_idx']} — debug view")

    # carré dyadique
    ax.add_patch(Rectangle((x0, y0), size, size, edgecolor="lightgray", facecolor="none", linewidth=1.0))

    # droite L
    t_vals = np.linspace(-1, 1, 300)
    line = np.array([p + t * d for t in t_vals])
    ax.plot(line[:, 0], line[:, 1], "--k", linewidth=1)

    # rectangles R1 / R2
    for sign, color, label in [(-1, "orange", "R1"), (1, "green", "R2")]:
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

    # points avant/après p dans l’ordre induit
    before = np.array(ordered[:p_index])
    after = np.array(ordered[p_index + 1:])

    if len(before):
        ax.scatter(before[:, 0], before[:, 1], s=30, color="black", label="before p")
    if len(after):
        ax.scatter(after[:, 0], after[:, 1], s=30, color="cyan", label="after p")

    # pivot p
    ax.scatter([p[0]], [p[1]], s=90, color="red", label="pivot p")

    # chemin induit (optionnel : utile en debug local)
    xs, ys = zip(*ordered)
    ax.plot(xs, ys, color="blue", linewidth=1, label="induced order path")

    # backtrack choisi
    ax.plot([p[0], q1[0], q2[0]], [p[1], q1[1], q2[1]], color="cyan", linewidth=3, label="chosen backtrack")

    # texte explicatif
    ax.text(
        0.01, -0.10,
        f"p  = {tuple(np.round(p,4))}\n"
        f"q1 = {tuple(np.round(q1,4))} (R1)\n"
        f"q2 = {tuple(np.round(q2,4))} (R2)\n"
        f"theta={theta:.4f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top"
    )

    ax.legend(loc="upper right")



# ============================================================
# 6) “TABLEAU DE BORD” : parcourir les carrés et debugger
# ============================================================
# Utilité:
# - Après la collecte, tu veux pouvoir inspecter facilement chaque carré.
# - Ici: on affiche la vue globale, puis on ouvre une vue par carré trouvé.
# - (Pour t=1 c’est OK, pour t=3 ça fait beaucoup: mais c’est exactement ce que tu as demandé.)
def debug_all_squares(points, t, results, l=0.3, w=0.08):
    display_global_selected_triplets(points, t, results, show_squares=True)

    for sq_idx in sorted(results.keys()):
        st = results[sq_idx]
        visualize_state_in_square(points, st, l=l, w=w, title=f"Debug square #{sq_idx} (t={t})")

    plt.show()

# ============================================================
# MAIN (pipeline complet)
# ============================================================
if __name__ == "__main__":
    # ===== Étape 1: générer le carré unité M×M =====
    M = 32
    points = generate_grid_points(M)

    # Choix ordre (comme tu veux)
    # order, _ = zcurve_order(points)
    # order, _ = platzman_order(points)
    order, _ = hilbert_order(points)

    # ===== Étape 2: paramètres dyadiques =====
    t = 1  # t=1 => 4 dyadic squares, t=2 => 16, etc.

    # paramètres rectangles
    l = 0.30
    w = 0.08

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

    # ===== Étape 4 + debug détaillé =====
    # - affiche globalement tous les triplets
    # - puis ouvre une vue debug pour chaque carré qui a un backtrack
    debug_all_squares(points, t, results, l=l, w=w)

    # ===== Étape suivante (pas maintenant) =====
    # tracer le chemin global sur les points sélectionnés, etc.

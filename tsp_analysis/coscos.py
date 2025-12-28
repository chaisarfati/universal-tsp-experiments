import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from math import cos, sin, pi, log2
from .geometry import generate_grid_points
from .heuristics import platzman_order, zcurve_order, hilbert_order
from .experiment import save_result_to_file
import random
from matplotlib.widgets import Button


def generate_dyadic_squares(scale: int):
    """Returns all dyadic squares of scale t."""
    step = 1 / (2 ** scale)
    squares = []
    for i in range(2 ** scale):
        for j in range(2 ** scale):
            x0 = i * step
            y0 = j * step
            squares.append(((x0, y0), step))
    return squares


def generate_directions(M):
    """Génère les M directions du papier (orientations de L)."""
    return [(cos(2 * pi * i / M), sin(2 * pi * i / M)) for i in range(1, M + 1)]


def project_point(p, direction):
    """Projette le point p sur la droite directionnelle (normée)."""
    return np.dot(p, direction)


def detect_backtracks_one_per_square_cosmas(
    points, order, scale=2, l=0.6, w=0.25, M=16, verbose=True
):
    directions = generate_directions(M)
    squares = generate_dyadic_squares(scale)

    ordered_points = [tuple(points[i]) for i in order]
    backtracks = []

    for sq_idx, ((x0, y0), size) in enumerate(squares):
        x1, y1 = x0 + size, y0 + size

        l_sq = l * size
        w_sq = w * size

        in_square = [
            p for p in ordered_points
            if x0 <= p[0] < x1 and y0 <= p[1] < y1
        ]

        if len(in_square) < 3:
            if verbose:
                print(f"[t={scale}] no backtrack in dyadic square #{sq_idx} (too few points)")
            continue

        found = False
        indices = list(range(len(in_square) - 2))
        random.shuffle(indices)

        for i in indices:
            if found:
                break

            p = np.array(in_square[i], dtype=float)

            for dir_vec in directions:
                dir_vec = np.array(dir_vec, dtype=float)
                dir_vec /= np.linalg.norm(dir_vec)
                ortho = np.array([-dir_vec[1], dir_vec[0]])

                q_plus = q_minus = None
                idx_plus = idx_minus = None

                for j, q0 in enumerate(in_square[i + 1:], start=i + 1):
                    q = np.array(q0, dtype=float)
                    v = q - p

                    tproj = np.dot(v, dir_vec)
                    uproj = np.dot(v, ortho)

                    if abs(uproj) > w_sq:
                        continue

                    if 0 < tproj <= l_sq and q_plus is None:
                        q_plus, idx_plus = q0, j
                    elif -l_sq <= tproj < 0 and q_minus is None:
                        q_minus, idx_minus = q0, j

                    if q_plus is not None and q_minus is not None:
                        # reconstruire le triplet dans l'ordre linéaire
                        candidates = [(idx_plus, q_plus), (idx_minus, q_minus), (i, tuple(p))]
                        candidates.sort(key=lambda x: x[0])
                        (ia, a), (ib, b), (ic, c) = candidates

                        # projections sur L
                        pa = project_point(np.array(a) - p, dir_vec)
                        pb = project_point(np.array(b) - p, dir_vec)
                        pc = project_point(np.array(c) - p, dir_vec)

                        # TEST CRUCIAL : retour en arrière au point médian
                        if not ((pa <= pb <= pc) or (pa >= pb >= pc)):
                            backtracks.append((a, b, c))
                            found = True
                            break

                if found:
                    break

        if not found and verbose:
            print(f"[t={scale}] no backtrack in dyadic square #{sq_idx}")

    return backtracks


def detect_backtracks_t0_correct(points, order, M=16):
    """
    Détection CORRECTE des backtracks pour t=0 uniquement.
    """
    points = np.array(points, float)
    ordered = [points[i] for i in order]
    directions = generate_directions(M)

    backtracks = []

    for i in range(len(ordered) - 2):
        a = ordered[i]
        b = ordered[i + 1]
        c = ordered[i + 2]

        for d in directions:
            d = np.array(d, float)
            d /= np.linalg.norm(d)

            pa = np.dot(a, d)
            pb = np.dot(b, d)
            pc = np.dot(c, d)

            if not (min(pa, pc) <= pb <= max(pa, pc)):
                backtracks.append((tuple(a), tuple(b), tuple(c)))
                return backtracks  # un seul suffit à t=0

    return backtracks


def find_best_scale_for_backtracks(
    points, order, max_scale, l=0.6, w=0.25, M=16
):
    results = {}

    for t in range(max_scale + 1):
        backtracks = detect_backtracks_one_per_square_cosmas(
            points, order,
            scale=t, l=l, w=w, M=M,
            verbose=False
        )

        # Nombre de carrés avec backtrack = nombre de backtracks
        results[t] = len(backtracks)

        print(f"[scale t={t}] backtracks found: {results[t]}")

    best_t = max(results, key=lambda t: results[t])

    print("\n=== SUMMARY ===")
    for t in sorted(results):
        print(f"t={t}: {results[t]} backtracks")

    print(f"\nBest scale: t={best_t} with {results[best_t]} backtracks")

    return best_t, results


def display_backtracks_graphically(points, order, scale=2, l=0.05, w=0.02, M=8, title=None):
    backtracks = detect_backtracks_one_per_square_cosmas(
        points, order, scale=scale, l=l, w=w, M=M
    )

    points = np.array(points, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)

    if title is None:
        ax.set_title(f"Cosmas Backtrack Detection (t={scale})")
    else:
        ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(False)

    for (x0, y0), size in generate_dyadic_squares(scale):
        ax.add_patch(
            plt.Rectangle((x0, y0), size, size,
                          linewidth=0.3, edgecolor='lightgray', facecolor='none')
        )

    ax.scatter(points[:, 0], points[:, 1], color='black', s=5, label='Points')

    path = points[order]
    segments = [[path[i], path[i + 1]] for i in range(len(path) - 1)]
    lc = LineCollection(segments, colors='blue', linewidths=1, label='Heuristic Path')
    ax.add_collection(lc)

    backtrack_lines = []
    for p1, p2, p3 in backtracks:
        for seg in [(p1, p2), (p2, p3)]:
            (x0, y0), (x1, y1) = seg
            line, = ax.plot([x0, x1], [y0, y1],
                            color='red', linewidth=2, label='Backtrack')
            backtrack_lines.append(line)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    ax.legend([h for h, l in zip(handles, labels) if not (l in seen or seen.add(l))],
              [l for l in labels if l not in seen])

    class ToggleBacktracks:
        def __init__(self):
            self.visible = True

        def toggle(self, event):
            self.visible = not self.visible
            for line in backtrack_lines:
                line.set_visible(self.visible)
            plt.draw()

    ax_button = plt.axes([0.3, 0.05, 0.4, 0.075])
    button = Button(ax_button, 'Afficher / Cacher Backtracks')
    button.on_clicked(ToggleBacktracks().toggle)

    plt.show()

def display_backtracks_for_all_scales(
    points, order, max_scale, l=0.6, w=0.25, M=16
):
    """
    Affiche une figure par échelle dyadique t = 0..max_scale
    avec les backtracks détectés (ou absence explicite).
    """
    for t in range(max_scale + 1):
        print(f"\n=== DISPLAY FOR t = {t} ===")

        display_backtracks_graphically(
            points,
            order,
            scale=t,
            l=l,
            w=w,
            M=M,
            title=f"Backtracks at dyadic scale t={t}"
        )

def display_single_backtrack_t0(points, order, l=0.6, w=0.25, M=16):
    """
    Sanity check absolu :
    - t = 0 (un seul dyadic square)
    - affiche UNIQUEMENT les 3 points du backtrack
    - bleu = rouge = ordre linéaire
    """
    backtracks = detect_backtracks_t0_correct(
        points,
        order,
        M=M    )

    if not backtracks:
        print("❌ No backtrack found at t=0")
        return

    # On prend le premier (il n'y en a qu'un max à t=0)
    p, q1, q2 = backtracks[0]

    # 🔴 CRITIQUE : retrouver l'ordre linéaire EXACT
    index = {tuple(points[i]): i for i in order}

    triplet = sorted([p, q1, q2], key=lambda x: index[tuple(x)])

    a, b, c = triplet

    print("Backtrack triplet in linear order:")
    print("a =", a)
    print("b =", b)
    print("c =", c)

    pts = np.array([a, b, c])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title("Sanity check t=0 — backtrack order")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Points
    ax.scatter(pts[:, 0], pts[:, 1], color="black", s=40)

    # Bleu : ordre linéaire
    ax.plot(pts[:, 0], pts[:, 1], color="blue", linewidth=2, label="Linear order")

    # Rouge : backtrack (DOIT être identique)
    ax.plot(pts[:, 0], pts[:, 1], color="red", linewidth=2, linestyle="--", label="Backtrack")

    ax.legend()
    plt.show()

if __name__ == "__main__":
    M = 16
    all_points = generate_grid_points(M)
    order, _ = zcurve_order(all_points)

    display_single_backtrack_t0(
        all_points,
        order,
        l=0.6,
        w=0.25,
        M=16
    )

# if __name__ == "__main__":
#     M = 16
#     n = M * M
#     r = int(log2(M))   # échelles raisonnables

#     all_points = generate_grid_points(M)
#     order, _ = platzman_order(all_points)

#     display_backtracks_for_all_scales(
#         all_points,
#         order,
#         max_scale=r,
#         l=0.6,
#         w=0.25,
#         M=16
#     )

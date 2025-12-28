import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from math import cos, sin, pi, log2
import numpy as np
import matplotlib.pyplot as plt
from .geometry import generate_grid_points
from .heuristics import platzman_order, zcurve_order, hilbert_order
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .experiment import save_result_to_file

def generate_dyadic_squares(scale: int):
    """Returns all dyadic squares of scale t."""
    step = 1 / (2 ** scale)
    squares = []
    for i in range(2 ** scale):
        for j in range(2 ** scale):
            x0 = i * step
            y0 = j * step
            squares.append(((x0, y0), step))  # bottom-left corner and size
    return squares


def generate_directions(M):
    """Génère les M directions du papier (orientations de L)."""
    return [(cos(2 * pi * i / M), sin(2 * pi * i / M)) for i in range(1, M + 1)]


def project_point(p, direction):
    """Projette le point p sur la droite directionnelle (normée)."""
    return np.dot(p, direction)


def detect_backtracks_cosmas(points, order, scale=2, l=0.05, w=0.02, M=8):
    """Retourne la liste des triplets de points constituant des backtracks selon Cosmas."""
    directions = generate_directions(M)
    backtracks = []

    squares = generate_dyadic_squares(scale)

    ordered_points = [points[i] for i in order]

    for (x0, y0), size in squares:
        x1, y1 = x0 + size, y0 + size

        # points dans ce carré
        in_square = [p for p in ordered_points if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]

        if len(in_square) < 3:
            continue

        for i, p in enumerate(in_square[:-2]):
            for dir_vec in directions:
                ortho = np.array([-dir_vec[1], dir_vec[0]])
                p_proj = project_point(p, dir_vec)

                strip_min = p_proj - l
                strip_max = p_proj + l

                R1 = []
                R2 = []

                for q in in_square[i+1:]:  # on ne considère que les q > p dans l'ordre
                    q_proj = project_point(q, dir_vec)
                    d = np.linalg.norm(q - p)
                    if strip_min <= q_proj <= strip_max and d > 0:
                        side = np.dot(q - p, ortho)
                        if abs(side) > w:
                            if side > 0:
                                R1.append(q)
                            else:
                                R2.append(q)

                if R1 and R2:
                    # backtrack trouvé : on prend p, q1 dans R1, q2 dans R2 (par exemple premiers)
                    backtracks.append((p, R1[0], R2[0]))
                    break  # on sort dès qu’on en trouve un pour ce carré

    return backtracks


import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection

def display_backtracks_graphically(points, order, scale=2, l=0.05, w=0.02, M=8):
    """Affiche la grille, les dyadic squares, le chemin heuristique, et surligne les backtracks détectés."""
    backtracks = detect_backtracks_cosmas(points, order, scale=scale, l=l, w=w, M=M)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)  # laisse de la place pour les boutons

    ax.set_title("Cosmas Backtrack Detection")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(False)

    # Grille dyadique
    for (x0, y0), size in generate_dyadic_squares(scale):
        ax.add_patch(plt.Rectangle((x0, y0), size, size,
                                   linewidth=0.3, edgecolor='lightgray', facecolor='none'))

    # Points
    ax.scatter(points[:, 0], points[:, 1], color='black', s=5, label='Points')

    # Chemin heuristique
    path = points[order]
    segments = [[path[i], path[i+1]] for i in range(len(path)-1)]
    lc = LineCollection(segments, colors='blue', linewidths=1, label='Heuristic Path', zorder=1)
    ax.add_collection(lc)

    # Backtracks
    backtrack_lines = []
    for p1, p2, p3 in backtracks:
        print(f"backtrack for {p1}, {p2}, {p3}")
        for seg in [(p1, p2), (p2, p3)]:
            (x0, y0), (x1, y1) = seg
            line, = ax.plot([x0, x1], [y0, y1], color='red', linewidth=2, zorder=3, label='Backtrack')
            backtrack_lines.append(line)

    # Éviter les doublons dans la légende
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    ax.legend([h for h, l in zip(handles, labels) if not (l in seen or seen.add(l))],
              [l for l in labels if l not in seen])

    # Bouton toggle
    class ToggleBacktracks:
        def __init__(self):
            self.visible = True

        def toggle(self, event):
            self.visible = not self.visible
            for line in backtrack_lines:
                line.set_visible(self.visible)
            plt.draw()

    toggle = ToggleBacktracks()
    ax_button = plt.axes([0.3, 0.05, 0.4, 0.075])
    button = Button(ax_button, 'Afficher / Cacher Backtracks')
    button.on_clicked(toggle.toggle)

    plt.show()



if __name__ == "__main__":
    # Extrait les points et l’ordre heuristique pour les passer à l’affichage
    M = 16
    r = int(log2(M) // 2)  # car chaque carré de taille 1/2^scale accueillera une configuration MxM
    all_points = generate_grid_points(M)
    order, _ = zcurve_order(all_points)
    display_backtracks_graphically(all_points, order, scale=r)
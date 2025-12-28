import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from math import cos, sin, pi
import random

import numpy as np
from matplotlib.patches import Rectangle
from math import cos, sin, pi
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from math import cos, sin, pi, log2
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


def visualize_random_L_and_rectangles(points, order, l=0.3, w=0.08):
    """
    Visualisation debug :
    - pivot p aléatoire
    - direction L aléatoire
    - rectangles R1 / R2
    - points avant p : noir
    - points après p : violet
    """

    points = np.array(points, float)
    ordered = [points[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.22)

    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Debug géométrique : L, R1, R2 + ordre linéaire")
    ax.grid(False)

    # éléments graphiques persistants
    line_L, = ax.plot([], [], "--", color="black", label="L")
    path_line, = ax.plot([], [], color="blue", linewidth=1, label="Linear order")

    scatter_before = ax.scatter([], [], s=25, color="black", label="before p")
    scatter_after = ax.scatter([], [], s=25, color="purple", label="after p")

    pivot_scatter = ax.scatter([], [], s=90, color="red", label="pivot p")

    rects = []

    ax.legend(loc="upper right")

    def redraw():
        nonlocal rects

        # nettoyer anciens rectangles
        for r in rects:
            r.remove()
        rects = []

        # pivot aléatoire (évite les extrêmes)
        p_index = random.randint(2, len(ordered) - 3)
        p = ordered[p_index]

        # direction aléatoire
        theta = random.uniform(0, pi)
        d = np.array([cos(theta), sin(theta)])
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])

        # droite L
        t_vals = np.linspace(-1, 1, 300)
        line = np.array([p + t * d for t in t_vals])
        line_L.set_data(line[:, 0], line[:, 1])

        # rectangles R1 et R2
        for sign, color, label in [(-1, "orange", "R1"), (1, "green", "R2")]:
            center = p + sign * (l / 2) * d
            corner = center - (l / 2) * d - (w / 2) * n

            rect = Rectangle(
                corner,
                width=l,
                height=w,
                angle=np.degrees(theta),
                edgecolor=color,
                facecolor=color,
                alpha=0.35,
                label=label
            )
            ax.add_patch(rect)
            rects.append(rect)

        # séparer les points selon l'ordre linéaire
        before = np.array(ordered[:p_index])
        after = np.array(ordered[p_index + 1:])

        if len(before) > 0:
            scatter_before.set_offsets(before)
        else:
            scatter_before.set_offsets([])

        if len(after) > 0:
            scatter_after.set_offsets(after)
        else:
            scatter_after.set_offsets([])

        # chemin linéaire
        xs, ys = zip(*ordered)
        path_line.set_data(xs, ys)

        # pivot
        pivot_scatter.set_offsets([p])

        fig.canvas.draw_idle()

    # bouton
    class RandomButton:
        def on_click(self, event):
            redraw()

    ax_button = plt.axes([0.25, 0.06, 0.5, 0.1])
    button = Button(ax_button, "Random (p, L)")
    button.on_clicked(RandomButton().on_click)

    # premier affichage
    redraw()
    plt.show()



if __name__ == "__main__":
    M = 16
    points = generate_grid_points(M)
    order, _ = hilbert_order(points)

    visualize_random_L_and_rectangles(
        points,
        order,
        l=0.3,
        w=0.08
    )
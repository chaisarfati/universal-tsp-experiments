import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from math import cos, sin, pi
import random


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
    Visualisation debug avancée :
    - pivot p aléatoire
    - direction L aléatoire
    - rectangles R1 / R2
    - points avant p : noir
    - points après p : violet
    - bouton pour sélectionner explicitement un backtrack (p,q1,q2)
    """

    points = np.array(points, float)
    ordered = [points[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.32)

    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Debug géométrique : L, R1, R2 + sélection backtrack")
    ax.grid(False)

    # éléments graphiques
    line_L, = ax.plot([], [], "--", color="black", label="L")
    path_line, = ax.plot([], [], color="blue", linewidth=1, label="Linear order")

    scatter_before = ax.scatter([], [], s=25, color="black", label="before p")
    scatter_after = ax.scatter([], [], s=25, color="cyan", label="after p")
    pivot_scatter = ax.scatter([], [], s=90, color="red", label="pivot p")

    backtrack_line, = ax.plot([], [], color="cyan", linewidth=3, label="chosen backtrack")

    rects = []

    info_text = ax.text(
        0.01, -0.18, "", transform=ax.transAxes,
        fontsize=9, verticalalignment="top"
    )

    ax.legend(loc="upper right")

    # état courant
    state = {}

    def redraw():
        nonlocal rects, state

        # nettoyer rectangles et backtrack
        for r in rects:
            r.remove()
        rects = []
        backtrack_line.set_data([], [])
        info_text.set_text("")

        # pivot aléatoire
        p_index = random.randint(2, len(ordered) - 3)
        p = ordered[p_index]

        # direction aléatoire
        theta = random.uniform(0, pi)
        d = np.array([cos(theta), sin(theta)])
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])

        state = {
            "p_index": p_index,
            "p": p,
            "d": d,
            "n": n
        }

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

        # points selon ordre
        before = np.array(ordered[:p_index])
        after = np.array(ordered[p_index + 1:])

        scatter_before.set_offsets(before if len(before) else [])
        scatter_after.set_offsets(after if len(after) else [])

        xs, ys = zip(*ordered)
        path_line.set_data(xs, ys)
        pivot_scatter.set_offsets([p])

        fig.canvas.draw_idle()

    def choose_backtrack(event):
        if not state:
            return

        p = state["p"]
        d = state["d"]
        n = state["n"]
        p_index = state["p_index"]

        candidates_R1 = []
        candidates_R2 = []

        for q in ordered[p_index + 1:]:
            v = q - p
            tproj = np.dot(v, d)
            uproj = np.dot(v, n)

            if abs(uproj) > w / 2:
                continue

            if -l <= tproj < 0:
                candidates_R1.append(q)
            elif 0 < tproj <= l:
                candidates_R2.append(q)

        if not candidates_R1 or not candidates_R2:
            info_text.set_text("No valid backtrack: missing point in R1 or R2")
            fig.canvas.draw_idle()
            return

        q1 = random.choice(candidates_R1)
        q2 = random.choice(candidates_R2)

        # tracer le backtrack
        xs = [p[0], q1[0], q2[0]]
        ys = [p[1], q1[1], q2[1]]
        backtrack_line.set_data(xs, ys)

        info_text.set_text(
            f"Backtrack (p,q1,q2):\n"
            f"p  = {tuple(np.round(p, 4))}\n"
            f"q1 = {tuple(np.round(q1, 4))}  (R1)\n"
            f"q2 = {tuple(np.round(q2, 4))}  (R2)"
        )

        fig.canvas.draw_idle()

    # boutons
    ax_button1 = plt.axes([0.15, 0.08, 0.3, 0.1])
    ax_button2 = plt.axes([0.55, 0.08, 0.3, 0.1])

    btn_random = Button(ax_button1, "Random (p, L)")
    btn_backtrack = Button(ax_button2, "Select backtrack")

    btn_random.on_clicked(lambda e: redraw())
    btn_backtrack.on_clicked(choose_backtrack)
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
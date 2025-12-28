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


def find_backtrack_state(points, order, l=0.3, w=0.08, angle_steps=180, max_tries=50):
    points = np.array(points, float)
    ordered = [points[i] for i in order]

    if len(ordered) < 3:
        return None

    angles = np.linspace(0, pi, angle_steps, endpoint=False)

    for _ in range(max_tries):
        p_index = random.randint(2, len(ordered) - 3)
        p = ordered[p_index]

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


def visualize_random_L_and_rectangles(points, order, l=0.3, w=0.08):
    points = np.array(points, float)
    ordered = [points[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.28)

    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Visualisation backtrack (logique factorisée)")
    ax.grid(False)

    line_L, = ax.plot([], [], "--k")
    path_line, = ax.plot([], [], color="blue", linewidth=1)

    scatter_before = ax.scatter([], [], s=25, color="black")
    scatter_after = ax.scatter([], [], s=25, color="cyan")
    pivot_scatter = ax.scatter([], [], s=90, color="red")

    backtrack_line, = ax.plot([], [], color="cyan", linewidth=3)

    rects = []
    info = ax.text(0.01, -0.22, "", transform=ax.transAxes, fontsize=9)

    def redraw(event=None):
        nonlocal rects
        for r in rects:
            r.remove()
        rects = []
        backtrack_line.set_data([], [])
        info.set_text("")

        state = find_backtrack_state(points, order, l, w)
        if state is None:
            info.set_text("No backtrack found.")
            fig.canvas.draw_idle()
            return

        p = state["p"]
        d = state["d"]
        n = state["n"]
        theta = state["theta"]
        p_index = state["p_index"]
        q1 = state["q1"]
        q2 = state["q2"]

        # droite L
        t_vals = np.linspace(-1, 1, 300)
        line = np.array([p + t * d for t in t_vals])
        line_L.set_data(line[:, 0], line[:, 1])

        # rectangles
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
            rects.append(rect)

        before = np.array(ordered[:p_index])
        after = np.array(ordered[p_index + 1:])

        scatter_before.set_offsets(before if len(before) else [])
        scatter_after.set_offsets(after if len(after) else [])
        pivot_scatter.set_offsets([p])

        xs, ys = zip(*ordered)
        path_line.set_data(xs, ys)

        backtrack_line.set_data(
            [p[0], q1[0], q2[0]],
            [p[1], q1[1], q2[1]]
        )

        info.set_text(
            f"p = {tuple(np.round(p,4))}\n"
            f"q1 = {tuple(np.round(q1,4))} (R1)\n"
            f"q2 = {tuple(np.round(q2,4))} (R2)"
        )

        fig.canvas.draw_idle()

    ax_button = plt.axes([0.3, 0.08, 0.4, 0.1])
    Button(ax_button, "Find backtrack").on_clicked(redraw)

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
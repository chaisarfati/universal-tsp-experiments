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


def visualize_L_and_rectangles(points, order, p_index,
                               theta=0.0, l=0.3, w=0.08):
    """
    Visualise la droite L, les rectangles R1 et R2,
    et le chemin linéaire.
    """

    points = np.array(points, float)
    ordered = [points[i] for i in order]

    p = ordered[p_index]

    # direction L
    d = np.array([cos(theta), sin(theta)])
    d /= np.linalg.norm(d)

    # direction orthogonale
    n = np.array([-d[1], d[0]])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Debug géométrique : L, R1, R2")

    # tracer la droite L
    t_vals = np.linspace(-1, 1, 200)
    line = np.array([p + t * d for t in t_vals])
    ax.plot(line[:, 0], line[:, 1], color="black", linestyle="--", label="L")

    # rectangles R1 et R2
    for sign, label, color in [(-1, "R1", "orange"), (1, "R2", "green")]:
        center = p + sign * (l / 2) * d
        corner = center - (l / 2) * d - (w / 2) * n

        rect = Rectangle(
            corner,
            width=l,
            height=w,
            angle=np.degrees(theta),
            edgecolor=color,
            facecolor=color,
            alpha=0.3,
            label=label
        )
        ax.add_patch(rect)

    # points
    xs, ys = zip(*ordered)
    ax.scatter(xs, ys, color="black", s=25)

    # chemin linéaire
    ax.plot(xs, ys, color="blue", linewidth=1, label="Linear order")

    # pivot
    ax.scatter([p[0]], [p[1]], color="red", s=80, label="pivot p")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    M = 16
    points = generate_grid_points(M)
    order, _ = zcurve_order(points)

    visualize_L_and_rectangles(
        points,
        order,
        p_index=10,      # choisis un pivot
        theta=pi/6,      # direction L
        l=0.3,
        w=0.08
    )

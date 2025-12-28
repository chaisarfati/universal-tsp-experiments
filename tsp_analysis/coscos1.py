import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from math import cos, sin, pi
import random

from .geometry import generate_grid_points
from .heuristics import zcurve_order, platzman_order, hilbert_order


# ============================================================
# Directions
# ============================================================

def generate_directions(M):
    return [(cos(2 * pi * i / M), sin(2 * pi * i / M)) for i in range(1, M + 1)]


# ============================================================
# CORRECT backtrack detection for t = 0
# ============================================================

def detect_all_backtracks_t0_angle(points, order):
    """
    Détection correcte des backtracks à t=0
    via le critère angulaire strict (< 90°).
    """
    points = np.array(points, float)
    ordered = [points[i] for i in order]

    backtracks = []

    for i in range(len(ordered) - 2):
        a = ordered[i]
        b = ordered[i + 1]
        c = ordered[i + 2]

        u = a - b
        v = c - b

        # produit scalaire
        dot = np.dot(u, v)

        # angle strictement aigu
        if dot > 0:
            backtracks.append((tuple(a), tuple(b), tuple(c)))

    return backtracks


# ============================================================
# Interactive viewer for ONE order
# ============================================================

def interactive_backtrack_viewer(points, order, order_name, M=16):
    backtracks = detect_all_backtracks_t0_angle(points, order)

    if not backtracks:
        print(f"No backtracks found for order {order_name}")
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    plt.subplots_adjust(bottom=0.25)

    ax.set_title(f"{order_name} — random backtrack (t=0)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    scatter = ax.scatter([], [], s=60, color="black")
    line_blue, = ax.plot([], [], color="blue", linewidth=2, label="Linear order")
    line_red, = ax.plot([], [], color="red", linewidth=2, linestyle="--", label="Backtrack")

    ax.legend()

    def draw_random_backtrack():
        a, b, c = random.choice(backtracks)
        pts = np.array([a, b, c])

        scatter.set_offsets(pts)
        line_blue.set_data(pts[:, 0], pts[:, 1])
        line_red.set_data(pts[:, 0], pts[:, 1])

        fig.canvas.draw_idle()

    # initial draw
    draw_random_backtrack()

    class RandomButton:
        def on_click(self, event):
            draw_random_backtrack()

    ax_button = plt.axes([0.25, 0.08, 0.5, 0.1])
    button = Button(ax_button, "Random backtrack")
    button.on_clicked(RandomButton().on_click)

    plt.show()


# ============================================================
# MAIN — 3 ORDERS, 3 FIGURES
# ============================================================

if __name__ == "__main__":
    M = 16
    points = generate_grid_points(M)

    orders = [
        ("Z-curve", zcurve_order(points)[0]),
        ("Platzman", platzman_order(points)[0]),
        ("Hilbert", hilbert_order(points)[0]),
    ]

    for name, order in orders:
        interactive_backtrack_viewer(points, order, name, M=16)

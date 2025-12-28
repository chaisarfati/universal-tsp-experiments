import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# -------------------------
# STATIC PLOT
# -------------------------
def plot_grid_with_points(points: np.ndarray, M: int, path_orders=None, labels=None, title: str = ""):
    """Plot grid + points + optional paths (static)."""
    plt.figure(figsize=(6, 6))
    # Grid
    for i in range(M + 1):
        plt.axhline(i / M, color='lightgray', linewidth=0.5)
        plt.axvline(i / M, color='lightgray', linewidth=0.5)

    # Points
    plt.scatter(points[:, 0], points[:, 1], color='red', zorder=5)

    # Paths
    if path_orders:
        colors = ['blue', 'green']
        for idx, order in enumerate(path_orders):
            ordered = points[order]
            lbl = labels[idx] if labels else f"path {idx}"
            plt.plot(ordered[:, 0], ordered[:, 1], '-o', color=colors[idx % len(colors)], label=lbl)

    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    if labels:
        plt.legend()
    plt.grid(False)
    plt.show()


# -------------------------
# INTERACTIVE PLOT (CHECKBOX TOGGLE)
# -------------------------
def plot_grid_with_points_interactive(points: np.ndarray, M: int, path_orders=None, labels=None, title: str = ""):
    """Plot with interactive checkboxes to show/hide each path."""
    plt.figure(figsize=(7, 7))

    # Grid
    for i in range(M + 1):
        plt.axhline(i / M, color='lightgray', linewidth=0.5)
        plt.axvline(i / M, color='lightgray', linewidth=0.5)

    # Points
    plt.scatter(points[:, 0], points[:, 1], color='red', zorder=5, label='Points')

    # Paths
    lines = []
    colors = plt.cm.tab10.colors
    if path_orders:
        for idx, order in enumerate(path_orders):
            ordered = points[order]
            color = colors[idx % len(colors)]
            lbl = labels[idx] if labels else f"path {idx}"
            line, = plt.plot(ordered[:, 0], ordered[:, 1], '-o', color=color, lw=1.5, ms=4, label=lbl)
            lines.append(line)

    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.grid(False)

    # Interactive legend
    if labels:
        plt.subplots_adjust(left=0.25)
        height = 0.04 * len(labels)
        rax = plt.axes([0.02, 0.5 - height / 2, 0.18, height])
        check = CheckButtons(rax, labels, [True] * len(labels))

        def toggle_visibility(label):
            index = labels.index(label)
            visible = not lines[index].get_visible()
            lines[index].set_visible(visible)
            plt.draw()

        check.on_clicked(toggle_visibility)

    plt.show()


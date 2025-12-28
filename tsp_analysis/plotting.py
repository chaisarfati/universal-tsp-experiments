# plotting.py
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)


# ============================================================
# Helpers bas niveau (plomberie Tk + Matplotlib)
# ============================================================
def make_tk_figure(parent, figsize=(7, 7), title=None):
    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)

    if title:
        ax.set_title(title)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()

    canvas.draw_idle()
    return fig, ax, canvas


# ============================================================
# Visualisation locale : carré dyadique + backtrack
# ============================================================
def visualize_state_in_square(points, state):
    points = np.asarray(points, float)

    win = tk.Toplevel()
    win.title("Local backtrack view")
    win.geometry("700x700")

    sq = state["_square"]
    t = sq["t"]
    sq_idx = sq["sq_idx"]

    from .geometry import generate_dyadic_squares, filter_points_in_square

    squares = generate_dyadic_squares(t)
    (x0, y0), size = squares[sq_idx]

    mask = filter_points_in_square(points, x0, y0, size)
    idx = np.where(mask)[0].tolist()
    ordered = points[idx]

    p = state["p"]
    p_index = state["p_index"]
    q1 = state["q1"]
    q2 = state["q2"]
    d = state["d"]
    n = state["n"]
    theta = state["theta"]

    l = state["_geom"]["l"]
    w = state["_geom"]["w"]

    fig, ax, canvas = make_tk_figure(win, figsize=(6, 6))

    ax.set_aspect("equal")
    ax.set_xlim(x0, x0 + size)
    ax.set_ylim(y0, y0 + size)
    ax.grid(False)

    ax.add_patch(
        Rectangle((x0, y0), size, size, edgecolor="gray", facecolor="none")
    )

    ts = np.linspace(-1, 1, 400)
    line = np.array([p + tt * d for tt in ts])
    ax.plot(line[:, 0], line[:, 1], "--k", label="local line $L_0$")

    for sign, color in [(-1, "orange"), (1, "green")]:
        center = p + sign * (l / 2) * d
        corner = center - (l / 2) * d - (w / 2) * n
        ax.add_patch(
            Rectangle(
                corner, l, w,
                angle=np.degrees(theta),
                facecolor=color,
                edgecolor=color,
                alpha=0.3
            )
        )

    before = ordered[:p_index]
    after = ordered[p_index + 1:]

    if len(before):
        ax.scatter(before[:, 0], before[:, 1], s=25, color="black", label="before p")
    if len(after):
        ax.scatter(after[:, 0], after[:, 1], s=25, color="cyan", label="after p")

    ax.scatter([p[0]], [p[1]], s=90, color="red", label="pivot p")
    ax.plot([p[0], q1[0], q2[0]],
            [p[1], q1[1], q2[1]],
            color="cyan", linewidth=3, label="backtrack")

    if len(ordered) >= 2:
        ax.plot(ordered[:, 0], ordered[:, 1],
                color="blue", linewidth=1, alpha=0.6, label="induced order")

    ax.legend()
    canvas.draw_idle()




def display_order_vs_tsp(points, order_path, tsp_path,
                         heuristic_cost, tsp_cost,
                         pivot_state_map):
    """
    Independent Matplotlib (Tk-safe)

    - shows induced order path and exact TSP path
    - hover on a pivot:
        * points before in induced order -> blue
        * points after -> red
        * hovered pivot -> gold
        * shows rank and scale t
    - click on a pivot opens its backtrack window
    """
    points = np.asarray(points, float)

    # ==================================================
    # Dedicate Tk Window
    # ==================================================
    win = tk.Toplevel()
    win.title("Cosmas – Order vs TSP")
    win.geometry("900x900")

    # ==================================================
    # Matplotlib figure
    # ==================================================
    fig = Figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Order path vs Exact TSP")

    # ==================================================
    # Costs (outside the [0,1]² graphics)
    # ==================================================
    fig.text(
        0.02, 0.96,
        f"order_cost = {heuristic_cost:.3f}\noptimal_tsp = {tsp_cost:.3f}",
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.9)
    )

    sc = ax.scatter(
        points[:, 0], points[:, 1],
        s=40, color="black",
        zorder=5, picker=True
    )

    # ==================================================
    # Paths
    # ==================================================
    order_line, = ax.plot(
        points[order_path, 0],
        points[order_path, 1],
        linewidth=2,
        label="Order path"
    )

    tsp_line, = ax.plot(
        points[tsp_path, 0],
        points[tsp_path, 1],
        linewidth=2,
        label="Exact TSP"
    )

    ax.legend()

    # ==================================================
    # Canvas Tk
    # ==================================================
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    toolbar = NavigationToolbar2Tk(canvas, win)
    toolbar.update()

    canvas.draw_idle()

    # ==================================================
    # Butons Show / Hide
    # ==================================================
    def toggle(line):
        line.set_visible(not line.get_visible())
        canvas.draw_idle()

    controls = tk.Frame(win)
    controls.pack(side="bottom", fill="x", pady=4)

    def toggle_order():
        order_line.set_visible(not order_line.get_visible())
        canvas.draw_idle()

    def toggle_tsp():
        tsp_line.set_visible(not tsp_line.get_visible())
        canvas.draw_idle()

    tk.Button(controls, text="Show / Hide Order Path",
            command=toggle_order).pack(side="left", padx=10)

    tk.Button(controls, text="Show / Hide TSP Path",
            command=toggle_tsp).pack(side="left", padx=10)

    # ==================================================
    # Induced order rank
    # ==================================================
    n = len(points)
    rank = np.empty(n, dtype=int)
    rank[order_path] = np.arange(n)

    # ==================================================
    # Hover text
    # ==================================================
    hover_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85)
    )
    hover_text.set_visible(False)

    # ==================================================
    # Hover handler
    # ==================================================
    def on_move(event):
        if event.inaxes != ax or event.x is None or event.y is None:
            return

        mx, my = event.x, event.y
        pts_disp = ax.transData.transform(points)

        dists = np.hypot(
            pts_disp[:, 0] - mx,
            pts_disp[:, 1] - my
        )

        i = np.argmin(dists)

        # seuil en pixels
        if dists[i] > 10:
            sc.set_color("black")
            hover_text.set_visible(False)
            canvas.draw_idle()
            return

        r = rank[i]

        colors = []
        for j in range(n):
            if j == i:
                colors.append("gold")
            elif rank[j] < r:
                colors.append("dodgerblue")
            else:
                colors.append("tomato")

        sc.set_color(colors)

        p = tuple(points[i])
        t = pivot_state_map[p]["_square"]["t"]

        hover_text.set_text(
            f"pivot rank = {r}\nscale t = {t}"
        )
        hover_text.set_visible(True)

        canvas.draw_idle()

    # ==================================================
    # Click handler -> local backtrack
    # ==================================================
    def on_pick(event):
        if event.ind is None or len(event.ind) == 0:
            return

        # si plusieurs points sont détectés, on prend le plus proche
        i = int(event.ind[0])

        p = tuple(points[i])
        if p not in pivot_state_map:
            return

        state = pivot_state_map[p]
        visualize_state_in_square(points, state)

    # ==================================================
    # Events
    # ==================================================
    canvas.mpl_connect("motion_notify_event", on_move)
    canvas.mpl_connect("pick_event", on_pick)

    canvas.draw_idle()


######################
######################
######################

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


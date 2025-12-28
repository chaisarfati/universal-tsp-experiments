import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import math

from .utils import save_cosmas_result_to_file, load_cosmas_result_from_file
from .geometry import (
    generate_grid_points,
    generate_dyadic_squares,
    filter_points_in_square,
    find_backtrack_state,
    generate_random_global_line,
    point_distance_to_line,
    scale_lw_with_t
)
from .heuristics import hilbert_order
from .tsp_solver import compute_path_cost, solve_tsp_with_lkh
import os


# ============================================================
# ORDRE RESTREINT
# ============================================================
def restrict_order_to_square(order, square_indices_set):
    return [i for i in order if i in square_indices_set]


def collect_pivots_multiscale(points, order, r, verbose=True):
    results = {}
    for t in range(r + 1):
        squares = generate_dyadic_squares(t)
        l, w = scale_lw_with_t(t)

        for sq_idx, ((x0, y0), size) in enumerate(squares):
            mask = filter_points_in_square(points, x0, y0, size)
            idx = np.where(mask)[0].tolist()
            if len(idx) < 3:
                continue

            square_order = restrict_order_to_square(order, set(idx))
            state = find_backtrack_state(
                points, square_order,
                l=l, w=w,
                angle_steps=180,
                max_tries=80
            )
            if state is None:
                if verbose: print(f"[t={t}] square #{sq_idx}: No backtrack was found")
                continue

            if verbose:
                print(f"[t={t}] square #{sq_idx}: Backtrack found")


            state["_square"] = {"t": t, "sq_idx": sq_idx}
            state["_geom"] = {"l": l, "w": w}
            results[(t, sq_idx)] = state

    return results


# ============================================================
# EXTRACTION DE S = PIVOTS DANS LE STRIP
# ============================================================
def extract_strip_pivots(results, global_line, delta):
    p0, d, n, _ = global_line
    S = []
    for st in results.values():
        p = np.array(st["p"])
        if point_distance_to_line(p, p0, n) <= delta:
            S.append(tuple(p))
    return np.array(sorted(set(S)), dtype=float)


def induced_order(global_points, global_order, subset_points):
    index_map = {tuple(p): i for i, p in enumerate(subset_points)}
    return np.array(
        [index_map[tuple(global_points[i])]
         for i in global_order
         if tuple(global_points[i]) in index_map],
        dtype=int
    )


# ============================================================
# VISUALISATION LOCALE : DYADIC SQUARE + BACKTRACK
# ============================================================
import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle


def visualize_state_in_square(points, state):
    """
    Visualisation locale d'un carré dyadique + backtrack (Tk-safe)
    """

    points = np.asarray(points, float)

    # ==================================================
    # Fenêtre Tk dédiée
    # ==================================================
    win = tk.Toplevel()
    win.title("Local backtrack view")
    win.geometry("700x700")

    # ==================================================
    # Extraction des infos géométriques
    # ==================================================
    sq = state["_square"]
    t = sq["t"]
    sq_idx = sq["sq_idx"]

    squares = generate_dyadic_squares(t)
    (x0, y0), size = squares[sq_idx]

    mask = filter_points_in_square(points, x0, y0, size)
    idx = np.where(mask)[0].tolist()
    square_order = restrict_order_to_square(list(range(len(points))), set(idx))
    ordered = points[square_order]

    p = state["p"]
    p_index = state["p_index"]
    q1 = state["q1"]
    q2 = state["q2"]
    d = state["d"]
    n = state["n"]
    theta = state["theta"]

    l = state["_geom"]["l"]
    w = state["_geom"]["w"]

    # ==================================================
    # Figure Matplotlib (OO API — PAS pyplot)
    # ==================================================
    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_aspect("equal")
    ax.set_xlim(x0, x0 + size)
    ax.set_ylim(y0, y0 + size)
    ax.set_title(f"Dyadic square #{sq_idx} (t={t})")
    ax.grid(False)

    # ==================================================
    # Carré dyadique
    # ==================================================
    ax.add_patch(
        Rectangle(
            (x0, y0), size, size,
            edgecolor="gray",
            facecolor="none"
        )
    )

    # ==================================================
    # Droite locale L0
    # ==================================================
    ts = np.linspace(-1, 1, 400)
    line = np.array([p + tt * d for tt in ts])
    ax.plot(line[:, 0], line[:, 1], "--k", label="local line $L_0$")

    # ==================================================
    # Rectangles R1, R2
    # ==================================================
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

    # ==================================================
    # Points
    # ==================================================
    before = np.array(ordered[:p_index])
    after = np.array(ordered[p_index + 1:])

    if len(before):
        ax.scatter(before[:, 0], before[:, 1],
                   s=25, color="black", label="before p")

    if len(after):
        ax.scatter(after[:, 0], after[:, 1],
                   s=25, color="cyan", label="after p")

    ax.scatter([p[0]], [p[1]],
               s=90, color="red", label="pivot p")

    ax.plot([p[0], q1[0], q2[0]],
            [p[1], q1[1], q2[1]],
            color="cyan", linewidth=3, label="backtrack")

    if len(ordered) >= 2:
        ax.plot(ordered[:, 0], ordered[:, 1],
                color="blue", linewidth=1,
                alpha=0.6, label="induced order")

    ax.legend()

    # ==================================================
    # Canvas Tk
    # ==================================================
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    toolbar = NavigationToolbar2Tk(canvas, win)
    toolbar.update()

    canvas.draw_idle()


import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button


def display_order_vs_tsp(points, order_path, tsp_path,
                         heuristic_cost, tsp_cost,
                         pivot_state_map):
    """
    Fenêtre Matplotlib indépendante (Tk-safe)

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
    # Fenêtre Tk dédiée (ISOLATION TOTALE)
    # ==================================================
    win = tk.Toplevel()
    win.title("Cosmas – Order vs TSP")
    win.geometry("900x900")

    # ==================================================
    # Figure Matplotlib (OO API — PAS pyplot)
    # ==================================================
    fig = Figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Order path vs Exact TSP")

    # ==================================================
    # Coûts (hors [0,1]²)
    # ==================================================
    fig.text(
        0.02, 0.96,
        f"order_cost = {heuristic_cost:.3f}\noptimal_tsp = {tsp_cost:.3f}",
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.9)
    )

    # ==================================================
    # Scatter des pivots (UN SEUL ARTISTE)
    # ==================================================
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
    # canvas = FigureCanvasTkAgg(fig, master=win)
    # canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==================================================
    # Boutons Show / Hide
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
    # Click handler → backtrack local
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
    # Connexions événements
    # ==================================================
    canvas.mpl_connect("motion_notify_event", on_move)
    canvas.mpl_connect("pick_event", on_pick)

    canvas.draw_idle()


# def display_order_vs_tsp(points, order_path, tsp_path,
#                          heuristic_cost, tsp_cost,
#                          pivot_state_map):
#     """
#     Fenêtre Matplotlib indépendante (Tk-safe)
#     """

#     # --------------------------------------------------
#     # Fenêtre Tk dédiée
#     # --------------------------------------------------
#     win = tk.Toplevel()
#     win.title("Cosmas – Order vs TSP")
#     win.geometry("900x900")

#     # --------------------------------------------------
#     # Figure Matplotlib ISOLÉE (pas de plt.*)
#     # --------------------------------------------------
#     fig = Figure(figsize=(7, 7), dpi=100)
#     ax = fig.add_subplot(111)

#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_aspect("equal")
#     ax.set_title("Order path vs Exact TSP")

#     # --------------------------------------------------
#     # Scatter
#     # --------------------------------------------------
#     sc = ax.scatter(
#         points[:, 0], points[:, 1],
#         s=40, color="black", zorder=5, picker=True
#     )

#     # --------------------------------------------------
#     # Paths
#     # --------------------------------------------------
#     order_line, = ax.plot(
#         points[order_path, 0],
#         points[order_path, 1],
#         linewidth=2, label="Order path"
#     )

#     tsp_line, = ax.plot(
#         points[tsp_path, 0],
#         points[tsp_path, 1],
#         linewidth=2, label="Exact TSP"
#     )

#     ax.legend()

#     # --------------------------------------------------
#     # Costs (hors [0,1]^2)
#     # --------------------------------------------------
#     fig.text(
#         0.02, 0.92,
#         f"order_cost = {heuristic_cost:.3f}\noptimal_tsp = {tsp_cost:.3f}",
#         fontsize=9,
#         bbox=dict(boxstyle="round", fc="white", alpha=0.9)
#     )

#     # --------------------------------------------------
#     # Canvas Tk
#     # --------------------------------------------------
#     canvas = FigureCanvasTkAgg(fig, master=win)
#     canvas.get_tk_widget().pack(fill="both", expand=True)

#     # --------------------------------------------------
#     # Boutons show/hide
#     # --------------------------------------------------
#     def toggle(line):
#         line.set_visible(not line.get_visible())
#         canvas.draw_idle()

#     btn_ax1 = fig.add_axes([0.15, 0.02, 0.3, 0.05])
#     btn_ax2 = fig.add_axes([0.55, 0.02, 0.3, 0.05])

#     Button(btn_ax1, "Show / Hide Order").on_clicked(lambda e: toggle(order_line))
#     Button(btn_ax2, "Show / Hide TSP").on_clicked(lambda e: toggle(tsp_line))

#     canvas.draw_idle()

# ============================================================
# VISUALISATION GLOBALE : ORDRE vs TSP + CLIC
# ============================================================
# def display_order_vs_tsp(points, order_path, tsp_path,
#                          heuristic_cost, tsp_cost,
#                          pivot_state_map):
#     """
#     Global interactive visualization:
#     - shows induced order path and exact TSP path
#     - hover on a pivot:
#         * points before in induced order -> blue
#         * points after -> red
#         * hovered pivot -> gold
#         * shows rank and scale t
#     - click on a pivot opens its backtrack window
#     """

#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_aspect("equal")
#     ax.set_title("Order path vs Exact TSP")
#     fig.text(
#         0.02, 0.96,
#         f"order_cost = {heuristic_cost:.3f}\noptimal_tsp = {tsp_cost:.3f}",
#         ha="left", va="top",
#         fontsize=9,
#         bbox=dict(boxstyle="round", fc="white", alpha=0.9)
#     )
#     # --------------------------------------------------
#     # Scatter of pivots (ONE artist only – important)
#     # --------------------------------------------------
#     sc = ax.scatter(
#         points[:, 0], points[:, 1],
#         s=40, color="black", zorder=5, picker=True
#     )

#     # --------------------------------------------------
#     # Order path and TSP path
#     # --------------------------------------------------
#     order_line, = ax.plot(
#         points[order_path, 0],
#         points[order_path, 1],
#         linewidth=2, label="Order path"
#     )

#     tsp_line, = ax.plot(
#         points[tsp_path, 0],
#         points[tsp_path, 1],
#         linewidth=2, label="Exact TSP"
#     )

#     ax.legend()

#     # --------------------------------------------------
#     # Buttons: show / hide paths
#     # --------------------------------------------------
#     class Toggle:
#         def __init__(self, line):
#             self.line = line
#             self.visible = True

#         def toggle(self, event):
#             self.visible = not self.visible
#             self.line.set_visible(self.visible)
#             fig.canvas.draw_idle()

#     btn1 = Button(plt.axes([0.1, 0.02, 0.35, 0.06]), "Show / Hide Order Path")
#     btn2 = Button(plt.axes([0.55, 0.02, 0.35, 0.06]), "Show / Hide TSP Path")

#     btn1.on_clicked(Toggle(order_line).toggle)
#     btn2.on_clicked(Toggle(tsp_line).toggle)

#     # --------------------------------------------------
#     # Induced order rank
#     # --------------------------------------------------
#     n = len(points)
#     rank = np.empty(n, dtype=int)
#     rank[order_path] = np.arange(n)

#     # --------------------------------------------------
#     # Hover text
#     # --------------------------------------------------
#     hover_text = ax.text(
#         0.02, 0.98, "",
#         transform=ax.transAxes,
#         va="top", ha="left",
#         fontsize=9,
#         bbox=dict(boxstyle="round", fc="white", alpha=0.85)
#     )
#     hover_text.set_visible(False)

#     # --------------------------------------------------
#     # Hover handler (motion)
#     # --------------------------------------------------
#     def on_move(event):
#         if event.inaxes != ax or event.x is None or event.y is None:
#             return

#         # positions écran de la souris
#         mx, my = event.x, event.y

#         # positions écran des points
#         pts_disp = ax.transData.transform(points)

#         dists = np.hypot(
#             pts_disp[:, 0] - mx,
#             pts_disp[:, 1] - my
#         )

#         i = np.argmin(dists)

#         # rayon de détection en pixels (TRÈS important)
#         if dists[i] > 10:   # 10 pixels = bon compromis
#             sc.set_color("black")
#             hover_text.set_visible(False)
#             fig.canvas.draw_idle()
#             return

#         r = rank[i]

#         colors = []
#         for j in range(len(points)):
#             if j == i:
#                 colors.append("gold")
#             elif rank[j] < r:
#                 colors.append("dodgerblue")
#             else:
#                 colors.append("tomato")

#         sc.set_color(colors)

#         p = tuple(points[i])
#         t = pivot_state_map[p]["_square"]["t"]

#         hover_text.set_text(
#             f"pivot rank = {r}\nscale t = {t}"
#         )
#         hover_text.set_visible(True)

#         fig.canvas.draw_idle()
#     # --------------------------------------------------
#     # Click handler (open backtrack)
#     # --------------------------------------------------
#     def on_pick(event):
#         ind = event.ind
#         if len(ind) == 0:
#             return
#         i = ind[0]
#         p = tuple(points[i])

#         if p not in pivot_state_map:
#             return

#         state = pivot_state_map[p]
#         visualize_state_in_square(points, state)

#     fig.canvas.mpl_connect("motion_notify_event", on_move)
#     fig.canvas.mpl_connect("pick_event", on_pick)


#     plt.show()


def display_global_selected_pivots(points,
                                   order,
                                   tsp_order,
                                   pivot_state_map,
                                   title=None):
    """
    Global interactive viewer (pivot-only version).

    - Displays pivot points only (one per dyadic square).
    - Displays two paths:
        * induced linear order
        * exact TSP path
    - Click on a pivot opens its dyadic square and backtrack view.
    - Two buttons:
        * Show / Hide order path
        * Show / Hide TSP path
    """

    points = np.asarray(points, float)

    # --------------------------------------------------
    # Figure + axis (same layout as display_global_selected_triplets)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    button_bottom = 0.02
    button_height = 0.08
    button_top = button_bottom + button_height

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=button_top + 0.025
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)
    ax.grid(False)

    ax.set_title(title or "Global pivots: order vs TSP")

    # --------------------------------------------------
    # Pivot points (cliquables)
    # --------------------------------------------------
    artist_to_state = {}
    pivot_points = np.array(list(pivot_state_map.keys()))

    for p in pivot_points:
        sc = ax.scatter(
            [p[0]], [p[1]],
            s=30,
            color="black",
            picker=True,
            zorder=4
        )
        artist_to_state[sc] = pivot_state_map[tuple(p)]

    # --------------------------------------------------
    # Induced linear order path
    # --------------------------------------------------
    order_path = points[order]
    order_line, = ax.plot(
        order_path[:, 0],
        order_path[:, 1],
        color="blue",
        linewidth=2.0,
        alpha=0.7,
        label="Induced order path"
    )

    # --------------------------------------------------
    # Exact TSP path
    # --------------------------------------------------
    tsp_path = points[tsp_order]
    tsp_line, = ax.plot(
        tsp_path[:, 0],
        tsp_path[:, 1],
        color="orange",
        linewidth=2.0,
        alpha=0.7,
        label="Exact TSP path"
    )

    ax.legend(loc="upper right")

    # --------------------------------------------------
    # Buttons (same philosophy as before)
    # --------------------------------------------------
    class Toggle:
        def __init__(self, artist):
            self.artist = artist
            self.visible = True

        def toggle(self, event):
            self.visible = not self.visible
            self.artist.set_visible(self.visible)
            fig.canvas.draw_idle()

    ax_btn1 = plt.axes([0.10, button_bottom, 0.35, button_height])
    ax_btn2 = plt.axes([0.55, button_bottom, 0.35, button_height])

    btn1 = Button(ax_btn1, "Show / Hide Order Path")
    btn2 = Button(ax_btn2, "Show / Hide TSP Path")

    btn1.on_clicked(Toggle(order_line).toggle)
    btn2.on_clicked(Toggle(tsp_line).toggle)

    # --------------------------------------------------
    # Click handler: open local backtrack view
    # --------------------------------------------------
    local_debug_fig = {"fig": None}

    def on_pick(event):
        artist = event.artist
        if artist not in artist_to_state:
            return

        state = artist_to_state[artist]

        if local_debug_fig["fig"] is not None:
            plt.close(local_debug_fig["fig"])

        fig_local = visualize_state_in_square(points, state)
        local_debug_fig["fig"] = fig_local

    fig.canvas.mpl_connect("pick_event", on_pick)

    # --------------------------------------------------
    # Clean close
    # --------------------------------------------------
    def on_close(event):
        plt.close("all")

    fig.canvas.mpl_connect("close_event", on_close)

    print("[DEBUG] Interactive pivot viewer ready")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    data = load_cosmas_result_from_file("results_cosmas/grid_M512_set_k67_heuristic_hilbert_ratio106,14.npz")

    display_order_vs_tsp(
        points=data["points"],
        order_path=data["heur_order"],
        tsp_path=data["tsp_order"],
        heuristic_cost=data["heuristic_cost"],
        tsp_cost=data["tsp_cost"],
        pivot_state_map=data["pivot_state_map"],
    )

    M = 512
    points = generate_grid_points(M)
    order, _ = hilbert_order(points)

    r = 5
    results = collect_pivots_multiscale(points, order, r)
    print(f"Total pivots found: {len(results)}")

    global_line = generate_random_global_line()
    delta = 0.03

    S = extract_strip_pivots(results, global_line, delta)

    pivot_state_map = {
        tuple(st["p"]): st
        for st in results.values()
        if tuple(st["p"]) in map(tuple, S)
    }

    print(f"|S| = {len(S)}")

    heuristic_indices = induced_order(points, order, S)
    heuristic_cost = compute_path_cost(S, heuristic_indices)

    tsp_indices = solve_tsp_with_lkh(S)
    tsp_cost = compute_path_cost(S, np.array(tsp_indices))

    print("\n=== RESULTS ON S ===")
    print(f"Order path cost : {heuristic_cost:.6f}")
    print(f"TSP cost        : {tsp_cost:.6f}")
    print(f"Ratio           : {heuristic_cost / tsp_cost:.6f}")

    # display_global_selected_pivots(
    #     points=S,
    #     order=heuristic_indices,
    #     tsp_order=np.array(tsp_indices),
    #     pivot_state_map=pivot_state_map,
    #     title="Global pivots inside strip: order vs TSP"
    # )

    print("Saving results")
    save_cosmas_result_to_file(
        S=S,
        heur_order=heuristic_indices,
        tsp_order=np.array(tsp_indices),
        heuristic_cost=heuristic_cost,
        tsp_cost=tsp_cost,
        pivot_state_map=pivot_state_map,
        M=M,
        oracle_name="hilbert"    # ou autre si tu testes plusieurs heuristiques
    )
    print("Saved results")

    display_order_vs_tsp(
        points=S,
        order_path=heuristic_indices,
        tsp_path=np.array(tsp_indices),
        heuristic_cost=heuristic_cost,
        tsp_cost=tsp_cost,
        pivot_state_map=pivot_state_map
    )

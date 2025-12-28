import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from math import cos, sin, pi
import random

# ====== adapte ces imports à ton projet ======
from .geometry import generate_grid_points
from .heuristics import zcurve_order, platzman_order, hilbert_order


# ============================================================
# Dyadic squares
# ============================================================

def generate_dyadic_squares(t: int):
    """Return list of dyadic squares at scale t: [((x0,y0), size), ...]."""
    step = 1.0 / (2 ** t)
    squares = []
    for i in range(2 ** t):
        for j in range(2 ** t):
            x0 = i * step
            y0 = j * step
            squares.append(((x0, y0), step))
    return squares


def points_in_square(points, x0, y0, size):
    """Boolean mask for points inside [x0,x0+size) x [y0,y0+size)."""
    x1, y1 = x0 + size, y0 + size
    pts = np.asarray(points)
    return (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)


# ============================================================
# Geometry helpers for (p, L) and rectangles R1/R2
# ============================================================

def unit_dir(theta):
    d = np.array([cos(theta), sin(theta)], dtype=float)
    nrm = np.linalg.norm(d)
    return d / nrm if nrm != 0 else np.array([1.0, 0.0])


def rect_candidates_after_p(ordered_points, p_index, p, d, n, l, w):
    """
    From points AFTER p (in the filtered order for the square),
    return candidates in R1 (tproj in [-l,0)) and R2 (tproj in (0,l]).
    Rect width is w (so |uproj| <= w/2).
    """
    R1, R2 = [], []
    for q in ordered_points[p_index + 1:]:
        v = q - p
        tproj = float(np.dot(v, d))
        uproj = float(np.dot(v, n))

        if abs(uproj) > (w / 2.0):
            continue

        if -l <= tproj < 0:
            R1.append(q)
        elif 0 < tproj <= l:
            R2.append(q)
    return R1, R2


def find_first_angle_with_backtrack(ordered_points, p_index, p, l, w, angles):
    """
    Scan angles (0..pi) and return the first theta that yields at least one
    point after p in R1 and at least one point after p in R2.
    """
    for theta in angles:
        d = unit_dir(theta)
        n = np.array([-d[1], d[0]], dtype=float)
        R1, R2 = rect_candidates_after_p(ordered_points, p_index, p, d, n, l, w)
        if len(R1) > 0 and len(R2) > 0:
            return theta, d, n, R1, R2
    return None, None, None, [], []


# ============================================================
# Global overview window
# ============================================================

def show_global_window(points, t, title="GLOBAL"):
    squares = generate_dyadic_squares(t)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"{title} — dyadic scale t={t}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(False)

    # grid
    for (x0, y0), size in squares:
        ax.add_patch(
            plt.Rectangle((x0, y0), size, size,
                          linewidth=0.6, edgecolor="lightgray", facecolor="none")
        )

    pts = np.asarray(points, float)
    ax.scatter(pts[:, 0], pts[:, 1], s=15, color="black")
    return fig, ax


# ============================================================
# Per-square viewer class
# ============================================================

class DyadicSquareViewer:
    def __init__(self, square_id, x0, y0, size, points_global, global_order,
                 l=0.3, w=0.08, angle_steps=180):
        """
        square_id: index of the dyadic square
        points_global: (N,2)
        global_order: list/array of indices defining linear order over ALL points
        l,w are in absolute coordinates (same unit square)
        angle_steps: number of angles in [0,pi)
        """
        self.square_id = square_id
        self.x0, self.y0, self.size = x0, y0, size
        self.l = float(l)
        self.w = float(w)

        self.points_global = np.asarray(points_global, float)
        self.global_order = list(global_order)

        # filter points belonging to this square, BUT keep global order
        mask = points_in_square(self.points_global, x0, y0, size)
        idx_in_square = set(np.where(mask)[0].tolist())
        self.square_order_indices = [i for i in self.global_order if i in idx_in_square]
        self.ordered = np.array([self.points_global[i] for i in self.square_order_indices], float)

        # angles to scan (deterministic)
        self.angles = np.linspace(0.0, pi, angle_steps, endpoint=False)

        # state
        self.p_index = None
        self.p = None
        self.theta = None
        self.d = None
        self.n = None
        self.R1 = []
        self.R2 = []

        # UI
        self.fig = None
        self.ax = None
        self.line_L = None
        self.path_line = None
        self.scatter_before = None
        self.scatter_after = None
        self.pivot_scatter = None
        self.rects = []
        self.info_text = None
        self.btn_new_pivot = None
        self.btn_find_angle = None

    def _clear_rects(self):
        for r in self.rects:
            try:
                r.remove()
            except Exception:
                pass
        self.rects = []

    def _draw_L(self):
        # Draw L through pivot p
        t_vals = np.linspace(-1, 1, 300)
        line = np.array([self.p + t * self.d for t in t_vals])
        self.line_L.set_data(line[:, 0], line[:, 1])

    def _draw_rects(self):
        self._clear_rects()
        for sign, color, label in [(-1, "orange", "R1"), (1, "green", "R2")]:
            center = self.p + sign * (self.l / 2.0) * self.d
            corner = center - (self.l / 2.0) * self.d - (self.w / 2.0) * self.n

            rect = Rectangle(
                corner,
                width=self.l,
                height=self.w,
                angle=np.degrees(self.theta),
                edgecolor=color,
                facecolor=color,
                alpha=0.35,
                label=label
            )
            self.ax.add_patch(rect)
            self.rects.append(rect)

    def _draw_points_and_path(self):
        # Before/after in *square order*
        if self.p_index is None:
            before = self.ordered
            after = np.zeros((0, 2))
        else:
            before = self.ordered[:self.p_index]
            after = self.ordered[self.p_index + 1:]

        self.scatter_before.set_offsets(before if len(before) else [])
        self.scatter_after.set_offsets(after if len(after) else [])

        if len(self.ordered) >= 1:
            xs, ys = self.ordered[:, 0], self.ordered[:, 1]
            self.path_line.set_data(xs, ys)
        else:
            self.path_line.set_data([], [])

        if self.p is not None:
            self.pivot_scatter.set_offsets([self.p])
        else:
            self.pivot_scatter.set_offsets([])

    def _choose_random_pivot(self):
        # Need at least 3 points to even define p with after-p points
        if len(self.ordered) < 3:
            self.p_index = None
            self.p = None
            return False

        # avoid extremes to have "after" candidates
        self.p_index = random.randint(0, len(self.ordered) - 3)  # ensure at least 2 points after p
        self.p = self.ordered[self.p_index]
        return True

    def _find_angle_for_current_pivot(self):
        theta, d, n, R1, R2 = find_first_angle_with_backtrack(
            self.ordered, self.p_index, self.p,
            l=self.l, w=self.w, angles=self.angles
        )
        self.theta, self.d, self.n, self.R1, self.R2 = theta, d, n, R1, R2
        return theta is not None

    def _update_info(self, msg):
        self.info_text.set_text(msg)

    def redraw(self, force_new_pivot=True):
        # If no points in this square, show message
        if len(self.ordered) == 0:
            self._clear_rects()
            self.line_L.set_data([], [])
            self.p = None
            self.p_index = None
            self._draw_points_and_path()
            self._update_info("No points in this dyadic square.")
            self.fig.canvas.draw_idle()
            return

        # If few points, can’t have backtrack
        if len(self.ordered) < 3:
            self._clear_rects()
            self.line_L.set_data([], [])
            self.p = None
            self.p_index = None
            self._draw_points_and_path()
            self._update_info(f"Too few points ({len(self.ordered)}) to form a backtrack.")
            self.fig.canvas.draw_idle()
            return

        if force_new_pivot:
            self._choose_random_pivot()

        ok = self._find_angle_for_current_pivot()

        if not ok:
            # no backtrack for this pivot over scanned angles
            self.theta = None
            self.d = None
            self.n = None
            self._clear_rects()
            self.line_L.set_data([], [])
            self._draw_points_and_path()
            self._update_info(
                f"Square #{self.square_id}: pivot p picked, but NO backtrack found\n"
                f"(scanned {len(self.angles)} angles). Try another pivot."
            )
            self.fig.canvas.draw_idle()
            return

        # draw visuals
        self._draw_L()
        self._draw_rects()
        self._draw_points_and_path()

        self._update_info(
            f"Square #{self.square_id} | points={len(self.ordered)}\n"
            f"pivot index (in-square order)={self.p_index}, p={tuple(np.round(self.p, 4))}\n"
            f"found theta={self.theta:.3f} rad | |R1_after|={len(self.R1)} | |R2_after|={len(self.R2)}\n"
            f"l={self.l}, w={self.w}"
        )

        self.fig.canvas.draw_idle()

    def on_new_pivot(self, event):
        self.redraw(force_new_pivot=True)

    def on_find_angle(self, event):
        # keep pivot, re-scan angles
        self.redraw(force_new_pivot=False)

    def show(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.30)

        self.ax.set_aspect("equal")
        self.ax.set_xlim(self.x0, self.x0 + self.size)
        self.ax.set_ylim(self.y0, self.y0 + self.size)
        self.ax.set_title(f"Dyadic square #{self.square_id} (t>0)")

        # draw square boundary
        self.ax.add_patch(
            plt.Rectangle((self.x0, self.y0), self.size, self.size,
                          linewidth=1.0, edgecolor="lightgray", facecolor="none")
        )

        self.line_L, = self.ax.plot([], [], "--", color="black", label="L")
        self.path_line, = self.ax.plot([], [], color="blue", linewidth=1, label="order (in-square)")

        self.scatter_before = self.ax.scatter([], [], s=25, color="black", label="before p")
        self.scatter_after = self.ax.scatter([], [], s=25, color="cyan", label="after p")
        self.pivot_scatter = self.ax.scatter([], [], s=90, color="red", label="pivot p")

        self.info_text = self.ax.text(
            0.01, -0.22, "", transform=self.ax.transAxes,
            fontsize=9, verticalalignment="top"
        )

        self.ax.legend(loc="upper right")

        # Buttons
        ax_b1 = plt.axes([0.12, 0.08, 0.34, 0.10])
        ax_b2 = plt.axes([0.54, 0.08, 0.34, 0.10])

        self.btn_new_pivot = Button(ax_b1, "New pivot (p)")
        self.btn_find_angle = Button(ax_b2, "Find angle (theta)")

        # Critical: keep references so they stay clickable
        self.btn_new_pivot.on_clicked(self.on_new_pivot)
        self.btn_find_angle.on_clicked(self.on_find_angle)

        # initial draw
        self.redraw(force_new_pivot=True)

        return self.fig


# ============================================================
# Driver: global + per-square windows
# ============================================================

def run_dyadic_debug(points, order, t=1, l=0.3, w=0.08, angle_steps=180):
    # 1) global window
    show_global_window(points, t, title="GLOBAL")

    # 2) per-square windows
    squares = generate_dyadic_squares(t)
    viewers = []
    for sq_idx, ((x0, y0), size) in enumerate(squares):
        v = DyadicSquareViewer(
            square_id=sq_idx,
            x0=x0, y0=y0, size=size,
            points_global=points,
            global_order=order,
            l=l, w=w,
            angle_steps=angle_steps
        )
        v.show()
        viewers.append(v)

    # show all figures
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    M = 16
    points = generate_grid_points(M)

    # Choose the order you want to debug
    # order, _ = zcurve_order(points)
    # order, _ = platzman_order(points)
    order, _ = hilbert_order(points)

    # Dyadic scale t
    t = 1  # t=1 -> 4 squares, t=2 -> 16 squares, etc.

    # l,w in absolute coordinates (unit square)
    run_dyadic_debug(
        points=points,
        order=order,
        t=t,
        l=0.30,
        w=0.08,
        angle_steps=180  # scan 180 angles in [0,pi)
    )

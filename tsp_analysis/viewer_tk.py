import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import CheckButtons

from .experiment import load_result_from_file

def index_available_files(folder="results"):
    """
    Build a hierarchy of available files:
    data[M][heuristic] = sorted list of k values
    """
    data = {}
    if not os.path.exists(folder):
        return data

    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    for fname in files:
        parts = fname.replace(".npz", "").split("_")
        try:
            M = int(parts[1][1:])
            k = int(parts[3][1:])
            heuristic = parts[-1]
            data.setdefault(M, {}).setdefault(heuristic, set()).add(k)
        except Exception:
            continue

    for M in data:
        for h in data[M]:
            data[M][h] = sorted(list(data[M][h]))
    return data


class ResultsViewer(tk.Tk):
    """Tkinter GUI embedding an interactive Matplotlib figure."""
    def __init__(self, folder="results"):
        super().__init__()
        self.title("TSP Results Viewer")
        self.geometry("1250x950")
        self.folder = folder
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load index of files
        self.data_index = index_available_files(folder)
        if not self.data_index:
            messagebox.showwarning("Warning", f"No .npz files found in {folder}")
            self.destroy()
            return

        self.available_M = sorted(self.data_index.keys())
        self.available_heuristics = sorted({h for M in self.data_index for h in self.data_index[M]})

        # Tk variables
        self.var_M = tk.IntVar(value=self.available_M[0])
        self.var_h = tk.StringVar(value=self.available_heuristics[0])
        self.var_k = tk.IntVar()

        # Controls
        frame_controls = ttk.Frame(self)
        frame_controls.pack(side="top", fill="x", pady=10)

        ttk.Label(frame_controls, text="Grid (M):").grid(row=0, column=0, padx=5)
        self.combo_M = ttk.Combobox(frame_controls, textvariable=self.var_M,
                                    values=self.available_M, state="readonly", width=10)
        self.combo_M.grid(row=0, column=1)

        ttk.Label(frame_controls, text="Heuristic:").grid(row=0, column=2, padx=5)
        self.combo_h = ttk.Combobox(frame_controls, textvariable=self.var_h,
                                    values=self.available_heuristics, state="readonly", width=15)
        self.combo_h.grid(row=0, column=3)

        ttk.Label(frame_controls, text="Set size (k):").grid(row=0, column=4, padx=5)
        self.combo_k = ttk.Combobox(frame_controls, textvariable=self.var_k,
                                    state="readonly", width=10)
        self.combo_k.grid(row=0, column=5)

        ttk.Button(frame_controls, text="Show", command=self.show_selected_result).grid(row=0, column=6, padx=10)
        ttk.Button(frame_controls, text="Generate New", command=self.open_generation_dialog).grid(row=0, column=7, padx=10)
        ttk.Button(frame_controls, text="Recompute TSP", command=self.recompute_paths).grid(row=0, column=8, padx=10)

        # Matplotlib figure inside Tkinter
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bind events
        self.combo_M.bind("<<ComboboxSelected>>", lambda e: self.update_k_options())
        self.combo_h.bind("<<ComboboxSelected>>", lambda e: self.update_k_options())
        self.combo_k.bind("<<ComboboxSelected>>", lambda e: self.show_selected_result())

        # Interactivité : point glissable
        self._draggable_point_idx = None
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)


        # Initialize ks and first plot
        self.update_k_options()

    def on_close(self):
        """Close cleanly and release the terminal."""
        self.destroy()
        self.quit()

    def on_click(self, event):
        if event.inaxes != self.ax or self.points is None:
            return

        threshold = 0.01  # tolérance pour attraper un point
        for i, (x, y) in enumerate(self.points):
            dx, dy = x - event.xdata, y - event.ydata
            if dx * dx + dy * dy < threshold ** 2:
                self._draggable_point_idx = i
                break

    def on_motion(self, event):
        if self._draggable_point_idx is None or event.inaxes != self.ax:
            return

        # Met à jour la position du point
        i = self._draggable_point_idx
        self.points[i] = [event.xdata, event.ydata]
        self.refresh_plot()

    def on_release(self, event):
        self._draggable_point_idx = None

    def refresh_plot(self):
        # Sauvegarde de l'état de visibilité avant le redraw
        previous_visibility = {label: line.get_visible() for label, line in zip(self.labels, self.lines)} if hasattr(self, "lines") else {}

        self.ax.clear()
        M = self.var_M.get()
        h = self.var_h.get()
        k = self.var_k.get()

        for i in range(M + 1):
            self.ax.axhline(i / M, color='lightgray', linewidth=0.5)
            self.ax.axvline(i / M, color='lightgray', linewidth=0.5)

        self.ax.scatter(self.points[:, 0], self.points[:, 1], color='red', zorder=5, label='Points')

        ordered_heur = self.points[self.heur_order]
        ordered_tsp = self.points[self.tsp_order]
        colors = plt.cm.tab10.colors

        line_heur, = self.ax.plot(ordered_heur[:, 0], ordered_heur[:, 1], '-o',
                                color=colors[0], lw=1.5, ms=4, label=f"{h} Order")
        line_tsp, = self.ax.plot(ordered_tsp[:, 0], ordered_tsp[:, 1], '-o',
                                color=colors[1], lw=1.5, ms=4, label="TSP optimal")

        self.lines = [line_heur, line_tsp]
        self.labels = [f"{h} Order", "TSP optimal"]

        # Restauration de la visibilité précédente
        for label, line in zip(self.labels, self.lines):
            if label in previous_visibility:
                line.set_visible(previous_visibility[label])

        title = f"M = {M}, k = {k}, heuristic = {h}\n" \
                f"Ratio = {self.ratio:.4f} | log₂(k) = {self.log_of_k:.4f} | cosmas = {self.cosmas:.4f}"

        self.ax.set_title(title)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.legend()
        self.ax.grid(False)

        self.fig.subplots_adjust(left=0.2)
        rax = self.fig.add_axes([0.02, 0.7, 0.15, 0.15])
        self.check = CheckButtons(rax, self.labels,
                                [line.get_visible() for line in self.lines])  # appliquer l’état restauré ici

        def toggle_visibility(label):
            index = self.labels.index(label)
            self.lines[index].set_visible(not self.lines[index].get_visible())
            self.canvas.draw_idle()

        self.check.on_clicked(toggle_visibility)
        self.canvas.draw_idle()

    def open_generation_dialog(self):
        """Ouvre une fenêtre pop-up pour choisir les paramètres de génération."""
        popup = tk.Toplevel(self)
        popup.title("Generate New Experiment")
        popup.geometry("400x300")
        popup.grab_set()  # bloque l'interaction avec la fenêtre principale

        # Paramètres initiaux
        M_var = tk.IntVar(value=self.var_M.get())
        k_var = tk.IntVar(value=self.var_k.get())
        h_var = tk.StringVar(value=self.var_h.get())
        iter_var = tk.IntVar(value=100)

        # Widgets
        tk.Label(popup, text="Grid size M:").pack(pady=4)
        tk.Entry(popup, textvariable=M_var).pack()

        tk.Label(popup, text="Subset size k:").pack(pady=4)
        tk.Entry(popup, textvariable=k_var).pack()

        tk.Label(popup, text="Heuristic:").pack(pady=4)
        heuristic_menu = ttk.Combobox(popup, textvariable=h_var, values=self.available_heuristics, state="readonly")
        heuristic_menu.pack()

        tk.Label(popup, text="Max iterations:").pack(pady=4)
        tk.Entry(popup, textvariable=iter_var).pack()

        def generate_and_close():
            from tsp_analysis.experiment import find_best_subset_randomized_generic
            from tsp_analysis.heuristics import heuristics_registry

            M = M_var.get()
            k = k_var.get()
            h = h_var.get()
            max_iter = iter_var.get()

            if h not in heuristics_registry:
                messagebox.showerror("Erreur", f"Heuristique inconnue : {h}")
                return

            try:
                find_best_subset_randomized_generic(
                    M=M,
                    k=k,
                    order_fn=[heuristics_registry[h]],
                    order_name=[h],
                    p=10,
                    seed=None,
                    max_iter=max_iter,
                )
                popup.destroy()
                self.data_index = index_available_files(self.folder)
                self.var_M.set(M)
                self.var_h.set(h)
                self.var_k.set(k)
                self.update_k_options()
                self.show_selected_result()
            except Exception as e:
                messagebox.showerror("Erreur de génération", str(e))

        tk.Button(popup, text="Generate", command=generate_and_close).pack(pady=10)


    def update_k_options(self):
        """Update available k values based on current (M, heuristic)."""
        M = self.var_M.get()
        h = self.var_h.get()

        ks = []
        if M in self.data_index and h in self.data_index[M]:
            ks = self.data_index[M][h]

        if not ks:
            self.combo_k["values"] = []
            self.var_k.set(0)
            messagebox.showinfo("Info", f"No files for M={M}, heuristic={h}")
            return

        self.combo_k["values"] = ks
        self.var_k.set(ks[0])
        self.show_selected_result()

    def show_selected_result(self):
        """Load the selected file and refresh the embedded interactive figure."""
        M = self.var_M.get()
        k = self.var_k.get()
        h = self.var_h.get()
        filename = f"{self.folder}/grid_M{M}_set_k{k}_heuristic_{h}.npz"

        if not os.path.exists(filename):
            return

        try:
            data = load_result_from_file(M, k, h, self.folder)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load {filename}:\n{e}")
            return

        self.points = data["points"]
        self.heur_order = data["heur_order"]
        self.tsp_order = data["tsp_order"]
        self.ratio = data["ratio"]
        self.log_of_k = data["log_of_k"]
        self.cosmas = data["cosmas"]

        self.ax.clear()

        # Grid
        for i in range(M + 1):
            self.ax.axhline(i / M, color='lightgray', linewidth=0.5)
            self.ax.axvline(i / M, color='lightgray', linewidth=0.5)

        # Points
        self.ax.scatter(self.points[:, 0], self.points[:, 1], color='red', zorder=5, label='Points')

        # Paths
        colors = plt.cm.tab10.colors
        ordered_heur = self.points[self.heur_order]
        ordered_tsp = self.points[self.tsp_order]

        line_heur, = self.ax.plot(
            ordered_heur[:, 0], ordered_heur[:, 1], '-o',
            color=colors[0], lw=1.5, ms=4, label=f"{h} Order"
        )
        line_tsp, = self.ax.plot(
            ordered_tsp[:, 0], ordered_tsp[:, 1], '-o',
            color=colors[1], lw=1.5, ms=4, label="TSP optimal"
        )

        self.lines = [line_heur, line_tsp]
        self.labels = [f"{h} Order", "TSP optimal"]

        title = (
            f"M = {M}, k = {k}, heuristic = {h}\n"
            f"Ratio = {self.ratio:.4f} | log₂(k) = {self.log_of_k:.4f} | cosmas = {self.cosmas:.4f}"
        )
        self.ax.set_title(title)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.grid(False)
        self.ax.legend()

        # Interactive checkboxes
        self.fig.subplots_adjust(left=0.2)
        rax = self.fig.add_axes([0.02, 0.7, 0.15, 0.15])
        self.check = CheckButtons(rax, self.labels, [True, True])

        def toggle_visibility(label):
            index = self.labels.index(label)
            line = self.lines[index]
            line.set_visible(not line.get_visible())
            self.canvas.draw_idle()

        self.check.on_clicked(toggle_visibility)

        self.canvas.draw_idle()

    def recompute_paths(self):
        """Recompute heuristic and TSP orders based on modified point positions."""
        from .heuristics import heuristics_registry
        from .tsp_solver import solve_tsp_with_lkh, compute_path_cost
        import math

        points = self.points
        h_name = self.var_h.get()

        if h_name not in heuristics_registry:
            messagebox.showerror("Error", f"Heuristic '{h_name}' not found in registry.")
            return

        heuristic_fn = heuristics_registry[h_name]
        try:
            heuristic_indices, _ = heuristic_fn(points)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply heuristic '{h_name}':\n{e}")
            return

        tsp_indices = solve_tsp_with_lkh(points)

        # Recompute metrics
        heuristic_cost = compute_path_cost(points, heuristic_indices)
        tsp_cost = compute_path_cost(points, tsp_indices)
        ratio = heuristic_cost / tsp_cost if tsp_cost > 0 else float("inf")
        log_of_k = math.log2(len(points)) if len(points) > 0 else 0
        cosmas = ratio / math.sqrt(len(points) / log_of_k) if log_of_k > 0 else 0

        # Update UI
        self.heur_order = heuristic_indices
        self.tsp_order = tsp_indices

        self.ax.clear()

        M = self.var_M.get()

        for i in range(M + 1):
            self.ax.axhline(i / M, color='lightgray', linewidth=0.5)
            self.ax.axvline(i / M, color='lightgray', linewidth=0.5)

        self.ax.scatter(points[:, 0], points[:, 1], color='red', zorder=5, label='Points')

        colors = plt.cm.tab10.colors
        ordered_heur = points[heuristic_indices]
        ordered_tsp = points[tsp_indices]

        line_heur, = self.ax.plot(
            ordered_heur[:, 0], ordered_heur[:, 1], '-o',
            color=colors[0], lw=1.5, ms=4, label=f"{h_name} Order"
        )
        line_tsp, = self.ax.plot(
            ordered_tsp[:, 0], ordered_tsp[:, 1], '-o',
            color=colors[1], lw=1.5, ms=4, label="TSP optimal"
        )

        self.lines = [line_heur, line_tsp]
        self.labels = [f"{h_name} Order", "TSP optimal"]

        title = (
            f"M = {M}, k = {len(points)}, heuristic = {h_name}\n"
            f"Ratio = {ratio:.4f} | log₂(k) = {log_of_k:.4f} | cosmas = {cosmas:.4f}"
        )
        self.ax.set_title(title)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.grid(False)
        self.ax.legend()

        self.fig.subplots_adjust(left=0.2)
        rax = self.fig.add_axes([0.02, 0.7, 0.15, 0.15])
        self.check = CheckButtons(rax, self.labels, [True, True])

        def toggle_visibility(label):
            index = self.labels.index(label)
            line = self.lines[index]
            line.set_visible(not line.get_visible())
            self.canvas.draw_idle()

        self.check.on_clicked(toggle_visibility)
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = ResultsViewer(folder="results")
    app.mainloop()


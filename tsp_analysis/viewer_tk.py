import os
import re
import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import CheckButtons
from .plotting import display_order_vs_tsp
from . import cosmas
from .heuristics import heuristics_registry_dict
from .utils import save_generic_utsp_result_to_file, load_generic_utsp_result_from_file, load_cosmas_result_from_file
from .plotting import display_order_vs_tsp
from .cosmas import (
    collect_pivots_multiscale,
    extract_strip_pivots,
)

# ============================================================
# Classic indexing (inchangé / compatible avec ton format actuel)
# ============================================================
def index_available_files(folder="results"):
    """
    Build a hierarchy of available files:
    data[M][heuristic] = sorted list of k values
    Filename pattern expected:
      grid_M{M}_set_k{k}_heuristic_{heuristic}.npz
    """
    data = {}
    if not os.path.exists(folder):
        return data

    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    for fname in files:
        parts = fname.replace(".npz", "").split("_")
        try:
            M = int(parts[1][1:])       # "M512" -> 512
            k = int(parts[3][1:])       # "k97" -> 97
            heuristic = parts[-1]       # last token
            data.setdefault(M, {}).setdefault(heuristic, set()).add(k)
        except Exception:
            continue

    for M in data:
        for h in data[M]:
            data[M][h] = sorted(list(data[M][h]))
    return data


# ================
# Cosmas indexing
# ================
_COSMAS_RE = re.compile(
    r"^grid_M(?P<M>\d+)_set_k(?P<k>\d+)_heuristic_(?P<h>.+?)(?:_.*)?\.npz$"
)

def index_available_cosmas_files(folder="results_cosmas"):
    """
    data[M][heuristic][k] = list of filenames (sorted)
    On parse le nom de fichier de manière robuste:
      grid_M512_set_k97_heuristic_hilbert_ratio86,32_ts20251225-153012.npz
      grid_M512_set_k97_heuristic_hilbert.npz
    etc.
    """
    data = {}
    if not os.path.exists(folder):
        return data

    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    for fname in files:
        m = _COSMAS_RE.match(fname)
        if not m:
            continue
        M = int(m.group("M"))
        k = int(m.group("k"))
        h = m.group("h")
        data.setdefault(M, {}).setdefault(h, {}).setdefault(k, []).append(fname)

    # tri stable
    for M in data:
        for h in data[M]:
            for k in data[M][h]:
                data[M][h][k] = sorted(data[M][h][k])
    return data

# ============================================================
# Tab 1 : Tab for Generic results 
# ============================================================
class ClassicResultsTab(ttk.Frame):
    """Ton viewer actuel, encapsulé dans un Frame (pour Notebook)."""
    def __init__(self, master, folder="results"):
        super().__init__(master)
        self.folder = folder

        # Load index of files
        self.data_index = index_available_files(folder)
        if not self.data_index:
            messagebox.showwarning("Warning", f"No .npz files found in {folder}")
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
        self.canvas.toolbar = None
        
        # Bind events
        self.combo_M.bind("<<ComboboxSelected>>", lambda e: self.update_k_options())
        self.combo_h.bind("<<ComboboxSelected>>", lambda e: self.update_k_options())
        self.combo_k.bind("<<ComboboxSelected>>", lambda e: self.show_selected_result())

        # Interactivité : point glissable
        self._draggable_point_idx = None
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.points = None
        self.heur_order = None
        self.tsp_order = None
        self.ratio = None
        self.log_of_k = None
        self.cosmas = None

        # Initialize
        self.update_k_options()

    def on_click(self, event):
        if event.inaxes != self.ax or self.points is None:
            return
        threshold = 0.01
        for i, (x, y) in enumerate(self.points):
            dx, dy = x - event.xdata, y - event.ydata
            if dx * dx + dy * dy < threshold ** 2:
                self._draggable_point_idx = i
                break

    def on_motion(self, event):
        if self._draggable_point_idx is None or event.inaxes != self.ax:
            return
        i = self._draggable_point_idx
        self.points[i] = [event.xdata, event.ydata]
        self.refresh_plot()

    def on_release(self, event):
        self._draggable_point_idx = None

    def refresh_plot(self):
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
                                [line.get_visible() for line in self.lines])

        def toggle_visibility(label):
            index = self.labels.index(label)
            self.lines[index].set_visible(not self.lines[index].get_visible())
            self.canvas.draw_idle()

        self.check.on_clicked(toggle_visibility)
        self.canvas.draw_idle()

    def open_generation_dialog(self):
        popup = tk.Toplevel(self)
        popup.title("Generate New Experiment")
        popup.geometry("400x300")
        popup.grab_set()

        M_var = tk.IntVar(value=self.var_M.get())
        k_var = tk.IntVar(value=max(self.var_k.get(), 10))
        h_var = tk.StringVar(value=self.var_h.get())
        iter_var = tk.IntVar(value=100)

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
            from tsp_analysis.heuristics import heuristics_registry_dict

            M = M_var.get()
            k = k_var.get()
            h = h_var.get()
            max_iter = iter_var.get()

            if h not in heuristics_registry_dict:
                messagebox.showerror("Erreur", f"Unknown heuristic : {h}")
                return

            try:
                find_best_subset_randomized_generic(
                    M=M,
                    k=k,
                    order_fn=[heuristics_registry_dict[h]],
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
                messagebox.showerror("Error in geneation", str(e))

        tk.Button(popup, text="Generate", command=generate_and_close).pack(pady=10)

    def update_k_options(self):
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
        M = self.var_M.get()
        k = self.var_k.get()
        h = self.var_h.get()
        filename = f"{self.folder}/grid_M{M}_set_k{k}_heuristic_{h}.npz"

        if not os.path.exists(filename):
            return

        try:
            data = load_generic_utsp_result_from_file(M, k, h, self.folder)
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

        for i in range(M + 1):
            self.ax.axhline(i / M, color='lightgray', linewidth=0.5)
            self.ax.axvline(i / M, color='lightgray', linewidth=0.5)

        self.ax.scatter(self.points[:, 0], self.points[:, 1], color='red', zorder=5, label='Points')

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
        from .heuristics import heuristics_registry_dict
        from .tsp_solver import solve_tsp_with_lkh, compute_path_cost
        from .save_utils import save_generic_utsp_result_to_file
        import math

        points = self.points
        h_name = self.h_var.get()   # ← heuristique sélectionnée
        M = self.M_var.get()        # ← grid size
        k = self.k_var.get()        # ← subset size

        if h_name not in heuristics_registry_dict:
            messagebox.showerror("Error", f"Heuristic '{h_name}' not found in registry.")
            return

        heuristic_fn = heuristics_registry_dict[h_name]

        try:
            heuristic_indices, _ = heuristic_fn(points)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to apply heuristic '{h_name}':\n{e}"
            )
            return

        tsp_indices = solve_tsp_with_lkh(points)

        heuristic_cost = compute_path_cost(points, heuristic_indices)
        tsp_cost = compute_path_cost(points, tsp_indices)

        if tsp_cost == 0:
            messagebox.showerror("Error", "TSP cost is zero, cannot compute ratio.")
            return

        ratio = heuristic_cost / tsp_cost

        # --- cosmas / log(k) (pour affichage) ---
        log2_k = math.log2(k) if k > 1 else 0
        cosmas = math.sqrt(log2_k / math.log2(log2_k)) if log2_k > 1 else float("nan")

        # --- stocker dans l'objet ---
        self.heur_order = heuristic_indices
        self.tsp_order = tsp_indices
        self.ratio = ratio
        self.log_of_k = log2_k
        self.cosmas = cosmas

        # --- sauvegarde (même contrat que le batch) ---
        save_generic_utsp_result_to_file(
            points=points,
            heur_order=heuristic_indices,
            tsp_order=tsp_indices,
            ratio=ratio,
            M=M,
            k=k,
            oracle_name=h_name,
            folder="results",
        )

        self.refresh_plot()


# ============================================================
# Tab 2 : Tab for Cosmas results
# ============================================================
class CosmasResultsTab(ttk.Frame):
    def __init__(self, master, folder="results_cosmas"):
        super().__init__(master)
        self.folder = folder

        # index
        self.data_index = index_available_cosmas_files(folder)

        # vars
        self.var_M = tk.IntVar(value=0)
        self.var_h = tk.StringVar(value="")
        self.var_k = tk.IntVar(value=0)
        self.var_file = tk.StringVar(value="")

        # Controls frame
        controls = ttk.Frame(self)
        controls.pack(side="top", fill="x", pady=10)

        ttk.Label(controls, text="Grid (M):").grid(row=0, column=0, padx=5, sticky="w")
        self.combo_M = ttk.Combobox(controls, textvariable=self.var_M, state="readonly", width=10)
        self.combo_M.grid(row=0, column=1, padx=5)

        ttk.Label(controls, text="Heuristic:").grid(row=0, column=2, padx=5, sticky="w")
        self.combo_h = ttk.Combobox(controls, textvariable=self.var_h, state="readonly", width=15)
        self.combo_h.grid(row=0, column=3, padx=5)

        ttk.Label(controls, text="|S| (k):").grid(row=0, column=4, padx=5, sticky="w")
        self.combo_k = ttk.Combobox(controls, textvariable=self.var_k, state="readonly", width=10)
        self.combo_k.grid(row=0, column=5, padx=5)

        ttk.Label(controls, text="Run file:").grid(row=0, column=6, padx=5, sticky="w")
        self.combo_file = ttk.Combobox(controls, textvariable=self.var_file, state="readonly", width=55)
        self.combo_file.grid(row=0, column=7, padx=5)

        ttk.Button(controls, text="Show", command=self.show_selected).grid(row=0, column=8, padx=10)
        ttk.Button(controls, text="Run Simulation", command=self.open_run_dialog).grid(row=0, column=9, padx=10)
        ttk.Button(controls, text="Refresh Index", command=self.refresh_index).grid(row=0, column=10, padx=10)

        # Info panel
        self.info = tk.Text(self, height=6, wrap="word")
        self.info.pack(fill="x", padx=10, pady=8)
        self.info.configure(state="disabled")

        # Bind updates
        self.combo_M.bind("<<ComboboxSelected>>", lambda e: self.update_h_options())
        self.combo_h.bind("<<ComboboxSelected>>", lambda e: self.update_k_options())
        self.combo_k.bind("<<ComboboxSelected>>", lambda e: self.update_file_options())
        self.combo_file.bind("<<ComboboxSelected>>", lambda e: self.preview_selected())

        # init combos
        self.refresh_index(init=True)

    def refresh_index(self, init=False):
        self.data_index = index_available_cosmas_files(self.folder)

        if not self.data_index:
            self._set_info(f"No .npz file found in {self.folder}\n"
                           f"You can click on 'Run Simulation' to generate a simulation and save it.")
            self.combo_M["values"] = []
            self.combo_h["values"] = []
            self.combo_k["values"] = []
            self.combo_file["values"] = []
            self.var_M.set(0)
            self.var_h.set("")
            self.var_k.set(0)
            self.var_file.set("")
            return

        Ms = sorted(self.data_index.keys())
        self.combo_M["values"] = Ms
        if init or self.var_M.get() not in Ms:
            self.var_M.set(Ms[0])

        self.update_h_options()

    def update_h_options(self):
        M = self.var_M.get()
        hs = sorted(self.data_index.get(M, {}).keys())
        self.combo_h["values"] = hs
        if not hs:
            self.var_h.set("")
            self.update_k_options()
            return
        if self.var_h.get() not in hs:
            self.var_h.set(hs[0])
        self.update_k_options()

    def update_k_options(self):
        M = self.var_M.get()
        h = self.var_h.get()
        ks = sorted(self.data_index.get(M, {}).get(h, {}).keys())
        self.combo_k["values"] = ks
        if not ks:
            self.var_k.set(0)
            self.update_file_options()
            return
        if self.var_k.get() not in ks:
            self.var_k.set(ks[0])
        self.update_file_options()

    def update_file_options(self):
        M = self.var_M.get()
        h = self.var_h.get()
        k = self.var_k.get()
        files = self.data_index.get(M, {}).get(h, {}).get(k, [])
        self.combo_file["values"] = files
        if not files:
            self.var_file.set("")
            self._set_info("No run for these parameters.")
            return
        if self.var_file.get() not in files:
            self.var_file.set(files[-1])  # dernier par défaut (souvent le plus récent si timestamp)
        self.preview_selected()

    def preview_selected(self):
        fname = self.var_file.get()
        if not fname:
            return
        path = os.path.join(self.folder, fname)
        try:
            d = load_cosmas_result_from_file(path)
        except Exception as e:
            self._set_info(f"Loading error: {e}")
            return

        msg = [
            f"File: {fname}",
            f"M = {d.get('M')} | |S| = {d.get('k')} | heuristic = {self.var_h.get()}",
        ]
        if d.get("heuristic_cost") is not None and d.get("tsp_cost") is not None:
            msg.append(f"heuristic_cost = {d['heuristic_cost']:.6f}")
            msg.append(f"tsp_cost       = {d['tsp_cost']:.6f}")
        if d.get("ratio") is not None:
            msg.append(f"ratio          = {d['ratio']:.6f}")
        if d.get("cosmas") is not None:
            msg.append(f"cosmas(|S|)    = {d['cosmas']:.6f}")
        if "ratio_over_cosmas" in d:
            msg.append(f"ratio/cosmas   = {d['ratio_over_cosmas']:.6f}")

        self._set_info("\n".join(msg))

    def show_selected(self):
        fname = self.var_file.get()
        if not fname:
            messagebox.showinfo("Info", "Choose a file to display.")
            return
        path = os.path.join(self.folder, fname)

        try:
            d = load_cosmas_result_from_file(path)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load {path}:\n{e}")
            return

        display_order_vs_tsp(
            points=d["points"],
            order_path=d["heur_order"],
            tsp_path=d["tsp_order"],
            heuristic_cost=d["heuristic_cost"],
            tsp_cost=d["tsp_cost"],
            pivot_state_map=d["pivot_state_map"],
        )

    def open_run_dialog(self):
        """
        Popup to launch a cosmas simulation  
        """
        popup = tk.Toplevel(self)
        popup.title("Run Cosmas Simulation")
        popup.geometry("430x360")
        popup.grab_set()

        M_var = tk.IntVar(value=self.var_M.get() if self.var_M.get() else 512)
        r_var = tk.IntVar(value=5)
        delta_var = tk.DoubleVar(value=0.03)
        oracle_var = tk.StringVar(value=self.var_h.get() if self.var_h.get() else "hilbert")

        tk.Label(popup, text="Grid size M:").pack(pady=6)
        tk.Entry(popup, textvariable=M_var).pack()

        tk.Label(popup, text="r (multiscale depth):").pack(pady=6)
        tk.Entry(popup, textvariable=r_var).pack()

        tk.Label(popup, text="delta (strip half-width):").pack(pady=6)
        tk.Entry(popup, textvariable=delta_var).pack()

        tk.Label(popup, text="oracle_name (heuristic label):").pack(pady=6)
        tk.Entry(popup, textvariable=oracle_var).pack()

        hint = tk.Label(
            popup,
            text="Note: The simulation can take a long time.",
            fg="gray"
        )
        hint.pack(pady=10)

        def run_and_close():
            try:
                from . import cosmas
            except Exception as e:
                messagebox.showerror("Import error", f"Impossible to import module cosmas.py\n{e}")
                return

            M = int(M_var.get())
            r = int(r_var.get())
            delta = float(delta_var.get())
            oracle_name = oracle_var.get().strip() or "hilbert"

            try:
                points = cosmas.generate_grid_points(M)
                # registry with normalized key
                heuristics_registry_ci = {
                    name.lower(): fn
                    for name, fn in heuristics_registry_dict.items()
                }
                oracle_key = oracle_name.strip().lower()
                if oracle_key not in heuristics_registry_ci:
                    messagebox.showerror("Error", f"Unknown heuristic '{oracle_name}'. "
                                    f"Available: {list(heuristics_registry_dict.keys())}")
                    return
                heuristic_fn = heuristics_registry_ci[oracle_key]

                # Generic call (compatible with all heuristics)
                order, _ = heuristic_fn(points)
                print(f"Calculated {oracle_key} order on point set, Now collecting backtrack pivots")
                results = cosmas.collect_pivots_multiscale(points, order, r)

                print(f"Finished computing backtracks. Now drawing random line")

                global_line = cosmas.generate_random_global_line()
                S = cosmas.extract_strip_pivots(results, global_line, delta)

                pivot_state_map = {
                    tuple(st["p"]): st
                    for st in results.values()
                    if tuple(st["p"]) in map(tuple, S)
                }


                heuristic_indices = cosmas.induced_order(points, order, S)
                heuristic_cost = cosmas.compute_path_cost(S, heuristic_indices)

                print(f"Computing optimal TSP with lk heuristics")
                tsp_indices = cosmas.solve_tsp_with_lkh(S)
                tsp_cost = cosmas.compute_path_cost(S, np.array(tsp_indices))

                # Sauvegarde (tu gères le nom dans save_cosmas_result_to_file)
                cosmas.save_cosmas_result_to_file(
                    S=S,
                    heur_order=np.array(heuristic_indices),
                    tsp_order=np.array(tsp_indices),
                    heuristic_cost=float(heuristic_cost),
                    tsp_cost=float(tsp_cost),
                    pivot_state_map=pivot_state_map,
                    M=M,
                    oracle_name=oracle_name,
                    folder=self.folder
                )

            except Exception as e:
                messagebox.showerror("Simulation error", str(e))
                return

            popup.destroy()
            self.refresh_index()
            messagebox.showinfo("OK", "Simulation completed and saved.")

        ttk.Button(popup, text="Run", command=run_and_close).pack(pady=14)

    def _set_info(self, text):
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)
        self.info.configure(state="disabled")


# ============================================================
# App principale avec Notebook
# ============================================================
class ResultsViewer(tk.Tk):
    def __init__(self, folder="results", cosmas_folder="results_cosmas"):
        super().__init__()
        self.title("TSP Results Viewer")
        self.geometry("1450x950")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.tab_classic = ClassicResultsTab(nb, folder=folder)
        self.tab_cosmas = CosmasResultsTab(nb, folder=cosmas_folder)

        nb.add(self.tab_classic, text="Classic results")
        nb.add(self.tab_cosmas, text="Cosmas results")

    def on_close(self):
        self.destroy()
        self.quit()


if __name__ == "__main__":
    app = ResultsViewer(folder="results", cosmas_folder="results_cosmas")
    app.mainloop()

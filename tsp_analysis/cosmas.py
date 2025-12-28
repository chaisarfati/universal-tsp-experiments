import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from .heuristics import hilbert_order
from .tsp_solver import compute_path_cost, solve_tsp_with_lkh
from .utils import (
    save_cosmas_result_to_file,
    load_cosmas_result_from_file,
)
from .geometry import (
    generate_grid_points,
    generate_dyadic_squares,
    filter_points_in_square,
    find_backtrack_state,
    generate_random_global_line,
    point_distance_to_line,
    scale_lw_with_t,
)
from .plotting import display_order_vs_tsp

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

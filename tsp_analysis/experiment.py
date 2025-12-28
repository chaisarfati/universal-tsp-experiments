import os
import math
import numpy as np

from .geometry import generate_grid_points
from .tsp_solver import solve_tsp_with_lkh, compute_path_cost
from .utils import pick_random_combs
from .plotting import plot_grid_with_points_interactive

# -------------------------
# SAVE / LOAD RESULTS
# -------------------------
def save_result_to_file(points, heur_order, tsp_order, ratio, M, k, oracle_name="heuristic", folder="results"):
    """Save one .npz per heuristic with metadata."""
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/grid_M{M}_set_k{k}_heuristic_{oracle_name}.npz"

    log_of_k = math.log2(k)
    cosmas = math.sqrt(log_of_k / math.log2(log_of_k))

    np.savez_compressed(
        filename,
        points=points,
        heur_order=heur_order,
        tsp_order=tsp_order,
        ratio=ratio,
        log_of_k=log_of_k,
        cosmas=cosmas,
    )
    print(f"💾 Saved: {filename}")

def load_result_from_file(M, k, oracle_name="hilbert", folder="results"):
    """Load a saved .npz result."""
    filename = f"{folder}/grid_M{M}_set_k{k}_heuristic_{oracle_name}.npz"
    data = np.load(filename)
    return {
        "points": data["points"],
        "heur_order": data["heur_order"],
        "tsp_order": data["tsp_order"],
        "log_of_k": data["log_of_k"],
        "cosmas": data["cosmas"],
        "ratio": data["ratio"].item()
    }


# -------------------------
# MAIN RANDOMIZED SEARCH
# -------------------------
def find_worst_subset_randomized_generic(
    M,
    k,
    order_fn,
    order_name="Order",
    max_iter=50,
    p=None,
    seed=None,
    verbose=True,
    folder="results",
):
    """
    Draw multiple random subsets; for each, compute heuristic/path ratio
    (Hilbert, Platzman, Z-order, etc.) while sharing a single TSP per subset.
    Saves one file per heuristic.
    """
    # Normalize inputs to lists
    if not isinstance(order_fn, (list, tuple)):
        order_fn = [order_fn]
    if not isinstance(order_name, (list, tuple)):
        order_name = [order_name]

    if seed is not None:
        np.random.seed(seed)

    # Generate full grid
    all_points = generate_grid_points(M)
    N = len(all_points)
    if verbose:
        print(f"✅ Grid generated with {N} points ({M}x{M})")

    # Init results per heuristic
    results = {
        name: {"worst_ratio": -np.inf, "worst_subset": None, "best_orders": None}
        for name in order_name
    }

    # Random combinations
    random_combs = pick_random_combs(N, k, max_iter, seed=seed)

    # Main loop
    for idx, indices in enumerate(random_combs):
        subset = all_points[list(indices)]

        # Solve TSP once per subset
        tsp_order = solve_tsp_with_lkh(subset)
        tsp_cost = compute_path_cost(subset, tsp_order)
        if tsp_cost == 0:
            continue

        # Evaluate all heuristics
        for fn, name in zip(order_fn, order_name):
            if fn.__name__ == "hilbert_order" and p is not None:
                heur_order, _ = fn(subset, p=p)
            else:
                heur_order, _ = fn(subset)

            heur_cost = compute_path_cost(subset, heur_order)
            ratio = heur_cost / tsp_cost

            if verbose and idx % max(1, max_iter // 10) == 0:
                print(f"[{name}] subset {idx}/{max_iter} | ratio {ratio:.4f} | best {results[name]['worst_ratio']:.4f}")

            # Update worst ratio
            if ratio > results[name]["worst_ratio"]:
                results[name]["worst_ratio"] = ratio
                results[name]["worst_subset"] = subset
                results[name]["best_orders"] = (heur_order, tsp_order)

    # Save results
    for name, r in results.items():
        if r["worst_subset"] is None:
            continue

        save_result_to_file(
            r["worst_subset"],
            r["best_orders"][0],
            r["best_orders"][1],
            r["worst_ratio"],
            M,
            k,
            oracle_name=name,
            folder=folder,
        )

        print(f"Saved result for {name}: ratio={r['worst_ratio']:.4f}")

    print("Search completed for all orders.")

def find_best_subset_randomized_generic(
    M,
    k,
    order_fn,
    order_name="Order",
    max_iter=50,
    p=None,
    seed=None,
    verbose=True,
    folder="results",
):
    """
    Draw multiple random subsets; for each, compute heuristic/path ratio
    (Hilbert, Platzman, Z-order, etc.) while sharing a single TSP per subset.
    Saves one file per heuristic.
    """
    # Normalize inputs to lists
    if not isinstance(order_fn, (list, tuple)):
        order_fn = [order_fn]
    if not isinstance(order_name, (list, tuple)):
        order_name = [order_name]

    if seed is not None:
        np.random.seed(seed)

    # Generate full grid
    all_points = generate_grid_points(M)
    N = len(all_points)
    if verbose:
        print(f"✅ Grid generated with {N} points ({M}x{M})")

    # Init results per heuristic
    results = {
        name: {"best_ratio": np.inf, "best_subset": None, "best_orders": None}
        for name in order_name
    }

    # Random combinations
    random_combs = pick_random_combs(N, k, max_iter, seed=seed)
    print(f"✅ random_combs : {random_combs}")


    # Main loop
    for idx, indices in enumerate(random_combs):
        subset = all_points[list(indices)]

        # Solve TSP once per subset
        tsp_order = solve_tsp_with_lkh(subset)
        tsp_cost = compute_path_cost(subset, tsp_order)

        if tsp_cost == 0:
            continue

        # Evaluate all heuristics
        for fn, name in zip(order_fn, order_name):
            if fn.__name__ == "hilbert_order" and p is not None:
                heur_order, _ = fn(subset, p=p)
            else:
                heur_order, _ = fn(subset)

            heur_cost = compute_path_cost(subset, heur_order)
            ratio = heur_cost / tsp_cost

            if verbose and idx % max(1, max_iter // 10) == 0:
                print(f"[{name}] subset {idx}/{max_iter} | ratio {ratio:.4f} | best {results[name]['best_ratio']:.4f}")

            # Update best ratio
            if ratio >= 1 and ratio < results[name]["best_ratio"]:
                results[name]["best_ratio"] = ratio
                results[name]["best_subset"] = subset
                results[name]["best_orders"] = (heur_order, tsp_order)

    # Save results
    for name, r in results.items():
        if r["best_subset"] is None:
            continue

        save_result_to_file(
            r["best_subset"],
            r["best_orders"][0],
            r["best_orders"][1],
            r["best_ratio"],
            M,
            k,
            oracle_name=name,
            folder=folder,
        )

        print(f"Saved result for {name}: ratio={r['best_ratio']:.4f}")

    print("Search completed for all orders.")


# -------------------------
# DEMO / RUNNERS
# -------------------------
def demo_load(M, k, oracle_name="Hilbert", folder="results"):
    """Display interactive plot for a saved result."""
    data = load_result_from_file(M, k, oracle_name, folder)
    points = data["points"]
    heur_order = data["heur_order"]
    tsp_order = data["tsp_order"]
    ratio = data["ratio"]
    log_of_k = data["log_of_k"]
    cosmas = data["cosmas"]

    title = (
        f"Heuristic vs TSP optimal\n"
        f"M = {M}, k = {k}, heuristic = {oracle_name}\n"
        f"Ratio = {ratio:.4f} | log₂(k) = {log_of_k:.4f} | cosmas = {cosmas:.4f}"
    )

    plot_grid_with_points_interactive(
        points,
        M,
        path_orders=[heur_order, tsp_order],
        labels=[f"{oracle_name} Order", "TSP optimal"],
        title=title
    )


def run_experiments():
    """Example loop over grids and subset sizes."""
    from .heuristics import zcurve_order, platzman_order, hilbert_order
    from .utils import max_iter_from_k

    for M in range(8, 33, 4):  # 8, 12, ..., 32
        total_points = M * M
        print(f"\n📦 Grid {M}x{M} ({total_points} points)")

        for percent in range(10, 100, 10):  # 10% to 100%
            k = (total_points * percent) // 100
            max_iter = max_iter_from_k(k)

            print(f"➡️ M={M}, k={k} ({percent}%), max_iter={max_iter}")

            find_best_subset_randomized_generic(
                M=M,
                k=k,
                order_fn=[zcurve_order, platzman_order, hilbert_order],
                order_name=["Z-order", "Platzman", "Hilbert"],
                p=10,
                seed=42,
                max_iter=max_iter,
            )

            print(f"✅ Done for M={M}, k={k}")


import numpy as np
import os
import math

# -------------------------
# RANDOM COMBINATIONS
# -------------------------
def pick_random_combs(N: int, k: int, max_iter: int, seed=None):
    """Generate up to max_iter unique random k-subsets from N elements."""
    if seed is not None:
        np.random.seed(seed)

    seen = set()
    combs = []
    attempts = 0

    while len(combs) < max_iter and attempts < 10 * max_iter:
        sample = tuple(sorted(np.random.choice(N, k, replace=False)))
        if sample not in seen:
            seen.add(sample)
            combs.append(np.array(sample))
        attempts += 1

    if len(combs) < max_iter:
        print(f"⚠️ Only {len(combs)} unique combinations generated out of requested {max_iter}")
    return combs


# -------------------------
# SMALL HELPER FOR EXPERIMENT BUDGET
# -------------------------
def max_iter_from_k(k: int) -> int:
    """Heuristic budget for number of random subsets vs k."""
    if k <= 8:
        return 1000
    elif k <= 10:
        return 500
    elif k <= 12:
        return 200
    elif k <= 15:
        return 50
    elif k <= 18:
        return 20
    elif k <= 20:
        return 10
    else:
        return 5

# -------------------------
# FILE SAVING UTILS
# -------------------------
def save_cosmas_result_to_file(
    S,
    heur_order,
    tsp_order,
    heuristic_cost,
    tsp_cost,
    pivot_state_map,
    M,
    oracle_name="heuristic",
    folder="results_cosmas"
):
    """
    Save full cosmas experiment result, including pivot states
    needed for interactive visualization.
    """
    os.makedirs(folder, exist_ok=True)

    n_S = len(S)
    if n_S < 4:
        raise ValueError("cosmas undefined for |S| < 4")

    log2_n = math.log2(n_S)
    cosmas = math.sqrt(log2_n / math.log2(log2_n))

    ratio = heuristic_cost / tsp_cost
    ratio_over_cosmas = ratio / cosmas

    # ratio en pourcentage, 2 décimales, virgule française
    ratio_pct = round(100 * ratio_over_cosmas, 2)
    ratio_str = f"{ratio_pct:.2f}".replace(".", ",")

    filename = (
        f"{folder}/"
        f"grid_M{M}_set_k{n_S}_heuristic_{oracle_name}_"
        f"ratio{ratio_str}.npz"
    )

    np.savez_compressed(
        filename,
        points=S,
        heur_order=heur_order,
        tsp_order=tsp_order,
        heuristic_cost=heuristic_cost,
        tsp_cost=tsp_cost,
        ratio=ratio,
        cosmas=cosmas,
        ratio_over_cosmas=ratio_over_cosmas,
        M=M,
        k=n_S,
        pivot_state_map=pivot_state_map,
    )

    print(f"💾 Saved: {filename}")

def load_cosmas_result_from_file(filename):
    data = np.load(filename, allow_pickle=True)
    return {
        "points": data["points"],
        "heur_order": data["heur_order"],
        "tsp_order": data["tsp_order"],
        "heuristic_cost": data["heuristic_cost"].item(),
        "tsp_cost": data["tsp_cost"].item(),
        "ratio": data["ratio"].item(),
        "cosmas": data["cosmas"].item(),
        "ratio_over_cosmas": data["ratio_over_cosmas"].item(),
        "M": int(data["M"]),
        "k": int(data["k"]),
        "pivot_state_map": data["pivot_state_map"].item(),
        "timestamp": data.get("timestamp", None),
    }

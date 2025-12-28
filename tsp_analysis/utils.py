import numpy as np

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


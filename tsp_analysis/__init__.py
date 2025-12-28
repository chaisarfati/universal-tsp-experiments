# Re-export convenient entry points for external use

from .geometry import (
    generate_grid_points,
    midpoint,
    triangle_centroid,
    sierpinski_sub,
    sierpinski_curve,
)

from .heuristics import (
    hilbert_order,
    zcurve_order,
    platzman_order,
)

from .tsp_solver import (
    write_tsplib_file,
    solve_tsp_with_lkh,
    compute_path_cost,
)

from .utils import (
    pick_random_combs,
    max_iter_from_k,
)

from .experiment import (
    find_worst_subset_randomized_generic,
    run_experiments,
)


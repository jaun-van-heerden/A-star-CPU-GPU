"""
lazy_astar.py — Lazy A* planner for robot arm configuration space.

Instead of precomputing the full O(n^d) C-space grid, this module runs A*
directly over joint-angle index space and calls the collision oracle only
for nodes actually expanded. This eliminates the exponential memory/compute
blow-up that makes dense-grid methods infeasible beyond 3-4 joints.

Angle convention
----------------
Local-relative with parent_angle - pi offset, matching aStarRobotArm_no_amin.py.
Each joint's angle index is scaled by step_int (degrees) and added to the
reversed direction of the parent arm. This is the physically correct model
for a serial-chain robot arm.

Required setup dict keys
------------------------
    arm_config      : list of dicts, each with 'length' (float) and 'angle-limit' (degrees)
    obstacle_config : list of (complex, complex) segment pairs  [optional, default []]
    step_int        : int  — degrees per index step
    deg_step        : int  — grid size per axis (must equal 360 // step_int)
"""

import heapq
import numpy as np


# ---------------------------------------------------------------------------
# Scalar geometry helpers
# ---------------------------------------------------------------------------

def _ccw(A, B, C):
    return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)


def _intersect(A, B, C, D):
    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)


def _calculate_segments(config, setup):
    """
    Compute arm segments for a single configuration.

    config : sequence of integer joint-angle indices, length d
    setup  : dict (see module docstring)

    Returns list of (start, end) complex pairs, length d.
    """
    start = complex(0, 0)
    parent_angle = 0.0
    segments = []

    for idx, arm in zip(config, setup["arm_config"]):
        angle_rad = parent_angle + setup["step_int"] * idx * (np.pi / 180.0)
        end = start + arm["length"] * np.exp(1j * angle_rad)
        segments.append((start, end))
        parent_angle = angle_rad - np.pi
        start = end

    return segments


def _is_collision(config, setup):
    """
    Return True if config self-intersects or hits an obstacle.
    Mirrors the logic in aStarRobotArm_no_amin.py: self_intersect + intersects_obstacle.
    """
    segments = _calculate_segments(config, setup)

    # Self-intersection (skip consecutive pairs)
    for i in range(len(segments) - 1):
        for j in range(i + 2, len(segments)):
            if _intersect(segments[i][0], segments[i][1],
                          segments[j][0], segments[j][1]):
                return True

    # Obstacle collision (skip first arm, matching original behaviour)
    for A, B in segments[1:]:
        for obs in setup.get("obstacle_config", []):
            if _intersect(A, B, obs[0], obs[1]):
                return True

    return False


# ---------------------------------------------------------------------------
# Vectorised geometry helpers  (plain NumPy, shape (N, d, 2) complex)
# ---------------------------------------------------------------------------

def _ccw_vec(A, B, C):
    return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)


def _calculate_segments_vec(configs, setup):
    """
    Vectorised segment computation.

    configs : np.ndarray shape (N, d), integer angle indices
    setup   : dict

    Returns segments : np.ndarray shape (N, d, 2), dtype complex128
    """
    N, d = configs.shape
    segments = np.zeros((N, d, 2), dtype=np.complex128)
    start = np.zeros(N, dtype=np.complex128)
    parent_angle = np.zeros(N, dtype=np.float64)

    lengths = np.array([arm["length"] for arm in setup["arm_config"]], dtype=np.float64)
    deg2rad = np.pi / 180.0

    for i, arm_length in enumerate(lengths):
        angle_rad = parent_angle + setup["step_int"] * configs[:, i].astype(np.float64) * deg2rad
        end = start + arm_length * np.exp(1j * angle_rad)
        segments[:, i, 0] = start
        segments[:, i, 1] = end
        parent_angle = angle_rad - np.pi
        start = end

    return segments


def _check_self_intersect_vec(segments):
    """
    segments : np.ndarray shape (N, d, 2), complex128

    Returns no_intersect : np.ndarray shape (N,), bool  (True = no self-intersection)
    """
    N, d, _ = segments.shape
    no_intersect = np.ones(N, dtype=bool)

    for i in range(d):
        for j in range(i + 2, d):
            A, B = segments[:, i, 0], segments[:, i, 1]
            C, D = segments[:, j, 0], segments[:, j, 1]
            cond1 = _ccw_vec(A, C, D) != _ccw_vec(B, C, D)
            cond2 = _ccw_vec(A, B, C) != _ccw_vec(A, B, D)
            no_intersect &= ~(cond1 & cond2)

    return no_intersect


def _check_obstacle_intersect_vec(segments, setup):
    """
    segments : np.ndarray shape (N, d, 2), complex128

    Returns no_obstacle : np.ndarray shape (N,), bool  (True = no obstacle hit)
    Skips arm 0, consistent with scalar _is_collision.
    """
    N, d, _ = segments.shape
    no_obstacle = np.ones(N, dtype=bool)
    obstacles = setup.get("obstacle_config", [])

    for j in range(1, d):
        A, B = segments[:, j, 0], segments[:, j, 1]
        for obs in obstacles:
            P, Q = obs[0], obs[1]
            cond1 = _ccw_vec(A, P, Q) != _ccw_vec(B, P, Q)
            cond2 = _ccw_vec(A, B, P) != _ccw_vec(A, B, Q)
            no_obstacle &= ~(cond1 & cond2)

    return no_obstacle


# ---------------------------------------------------------------------------
# Collision oracles
# ---------------------------------------------------------------------------

def _compute_bounds(setup):
    """Pre-compute per-joint (min_idx, max_idx) from angle-limit constraints."""
    deg_step = setup["deg_step"]
    bounds = []
    for arm in setup["arm_config"]:
        limit = arm["angle-limit"]
        min_idx = round((limit / 360.0) * deg_step)
        max_idx = deg_step - min_idx
        bounds.append((min_idx, max_idx))
    return bounds


class CollisionOracle:
    """
    Scalar collision oracle with memoisation.

    Checks one configuration at a time. Suitable for low-d spaces or when
    the batch overhead exceeds the geometry cost.
    """

    def __init__(self, setup: dict) -> None:
        self._setup = setup
        self._bounds = _compute_bounds(setup)
        self._cache: dict[tuple, bool] = {}
        self.oracle_calls = 0
        self.cache_hits = 0

    def _within_limits(self, config: tuple) -> bool:
        return all(lo <= idx <= hi for idx, (lo, hi) in zip(config, self._bounds))

    def is_valid(self, config: tuple) -> bool:
        """Return True if config is collision-free and within angle limits."""
        if config in self._cache:
            self.cache_hits += 1
            return self._cache[config]
        self.oracle_calls += 1
        result = self._within_limits(config) and not _is_collision(config, self._setup)
        self._cache[config] = result
        return result


class BatchedCollisionOracle:
    """
    Vectorised NumPy batch oracle.

    Checks N configurations in a single NumPy call. For axis-only A* with
    2*d neighbours per expansion this amortises Python-loop overhead and
    enables SIMD throughput in NumPy's C layer.
    """

    def __init__(self, setup: dict) -> None:
        self._setup = setup
        self._bounds = _compute_bounds(setup)
        self._bounds_arr = np.array(self._bounds, dtype=np.int32)  # shape (d, 2)
        self._cache: dict[tuple, bool] = {}
        self.oracle_calls = 0
        self.cache_hits = 0

    def _within_limits_vec(self, configs: np.ndarray) -> np.ndarray:
        """configs shape (N, d) → bool shape (N,)."""
        lo = self._bounds_arr[:, 0]  # (d,)
        hi = self._bounds_arr[:, 1]  # (d,)
        return np.all((configs >= lo) & (configs <= hi), axis=1)

    def is_valid_batch(self, configs: np.ndarray) -> np.ndarray:
        """
        configs : np.ndarray shape (N, d), integer angle indices
        Returns : np.ndarray shape (N,), bool
        """
        N = len(configs)
        result = np.zeros(N, dtype=bool)

        # Separate cache hits from misses
        miss_pos = []
        miss_cfgs = []
        for i, cfg in enumerate(configs):
            key = tuple(int(x) for x in cfg)
            if key in self._cache:
                self.cache_hits += 1
                result[i] = self._cache[key]
            else:
                miss_pos.append(i)
                miss_cfgs.append(cfg)

        if not miss_cfgs:
            return result

        miss_arr = np.array(miss_cfgs, dtype=np.int32)       # (M, d)
        self.oracle_calls += len(miss_arr)

        valid = self._within_limits_vec(miss_arr)             # (M,) bool

        # Only run geometry on angle-limit-passing configs
        geom_idx = np.where(valid)[0]
        if len(geom_idx) > 0:
            segs = _calculate_segments_vec(miss_arr[geom_idx], self._setup)
            no_self = _check_self_intersect_vec(segs)
            no_obs  = _check_obstacle_intersect_vec(segs, self._setup)
            valid[geom_idx] = no_self & no_obs

        for local_i, (orig_i, cfg) in enumerate(zip(miss_pos, miss_cfgs)):
            key = tuple(int(x) for x in cfg)
            v = bool(valid[local_i])
            self._cache[key] = v
            result[orig_i] = v

        return result

    def is_valid(self, config: tuple) -> bool:
        """Single-config convenience wrapper."""
        arr = np.array([list(config)], dtype=np.int32)
        return bool(self.is_valid_batch(arr)[0])


# ---------------------------------------------------------------------------
# Lazy A* solver
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("position", "cost", "total_cost")

    def __init__(self, position, cost: float, heuristic: float):
        self.position = position
        self.cost = cost
        self.total_cost = cost + heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost


class LazyAStarSolver:
    """
    A* planner over joint-angle index space that never precomputes C-space.

    The collision oracle is called only for nodes actually visited during
    search, replacing the O(n^d) precompute with a search-driven oracle
    that scales to any number of joints.

    Neighbours
    ----------
    Axis-only moves: change exactly one joint by ±1. This gives 2*d candidates
    per expansion (linear in d, not 3^d-1), keeps the heuristic admissible,
    and guarantees optimal paths under unit edge cost.

    API-compatible with AStarSolver from aStarXd.py.
    """

    def __init__(
        self,
        setup: dict,
        use_batched_oracle: bool = True,
        wrap_angles: bool = False,
        heuristic: str = "manhattan",
        weight: float = 1.0,
    ) -> None:
        """
        setup              : dict (see module docstring)
        use_batched_oracle : batch-check all neighbours in one NumPy call (default True)
        wrap_angles        : joint indices wrap at deg_step boundary (toroidal)
        heuristic          : "manhattan" (default, tighter) or "euclidean"
        weight             : > 1.0 trades optimality for speed (weighted A*)
        """
        assert setup["step_int"] * setup["deg_step"] == 360, (
            "setup['step_int'] * setup['deg_step'] must equal 360"
        )

        self._setup = setup
        self._wrap  = wrap_angles
        self._weight = weight
        self._ndim  = len(setup["arm_config"])
        self._n     = setup["deg_step"]

        if heuristic == "manhattan":
            self._h = lambda a, b: float(sum(abs(x - y) for x, y in zip(a, b)))
        else:
            self._h = lambda a, b: float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)

        if use_batched_oracle:
            self._oracle: BatchedCollisionOracle | CollisionOracle = BatchedCollisionOracle(setup)
        else:
            self._oracle = CollisionOracle(setup)
        self._batched = use_batched_oracle

        # Stats populated by solve()
        self._nodes_expanded = 0
        self._path_length: int | None = None

    # ------------------------------------------------------------------
    # Neighbour generation
    # ------------------------------------------------------------------

    def _neighbours(self, position: tuple) -> list:
        """
        Return up to 2*d axis-only neighbours (one joint ±1 at a time).
        Angle-limit pre-filtering is handled by the oracle; boundary
        clamping (or wrapping) is handled here.
        """
        pos = list(position)
        candidates = []
        for dim in range(self._ndim):
            for delta in (-1, +1):
                new_idx = pos[dim] + delta
                if self._wrap:
                    new_idx = new_idx % self._n
                elif not (0 <= new_idx < self._n):
                    continue
                nb = pos[:]
                nb[dim] = new_idx
                candidates.append(tuple(nb))
        return candidates

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, start: tuple, goal: tuple) -> "list[tuple] | None":
        """
        Return path as a list of config tuples (start … goal inclusive),
        or None if no path exists.

        Raises ValueError if start or goal is not a valid configuration.
        """
        if not self._oracle.is_valid(start):
            raise ValueError(f"Start configuration {start} is not a valid (collision-free) config.")
        if not self._oracle.is_valid(goal):
            raise ValueError(f"Goal configuration {goal} is not a valid (collision-free) config.")

        self._nodes_expanded = 0
        self._path_length = None

        open_heap  = [_Node(start, 0.0, self._weight * self._h(start, goal))]
        came_from: dict[tuple, _Node] = {}
        best_cost: dict[tuple, float] = {start: 0.0}
        visited:   set[tuple]         = set()

        while open_heap:
            current = heapq.heappop(open_heap)

            if current.position in visited:
                continue
            visited.add(current.position)
            self._nodes_expanded += 1

            if current.position == goal:
                path = [current.position]
                node = current
                while node.position in came_from:
                    node = came_from[node.position]
                    path.append(node.position)
                self._path_length = len(path)
                return path[::-1]

            candidates = self._neighbours(current.position)
            if not candidates:
                continue

            if self._batched:
                valid_mask = self._oracle.is_valid_batch(
                    np.array(candidates, dtype=np.int32)
                )
            else:
                valid_mask = [self._oracle.is_valid(c) for c in candidates]

            next_cost = current.cost + 1.0
            for neighbour, valid in zip(candidates, valid_mask):
                if not valid or neighbour in visited:
                    continue
                if next_cost < best_cost.get(neighbour, float("inf")):
                    best_cost[neighbour] = next_cost
                    came_from[neighbour] = current
                    h = self._weight * self._h(neighbour, goal)
                    heapq.heappush(open_heap, _Node(neighbour, next_cost, h))

        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return statistics from the most recent solve() call."""
        o = self._oracle
        return {
            "oracle_calls":   o.oracle_calls,
            "cache_hits":     o.cache_hits,
            "nodes_expanded": self._nodes_expanded,
            "path_length":    self._path_length,
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_setup = {
        "arm_config": [
            {"length": 1, "angle-limit": 0},
            {"length": 1, "angle-limit": 0},
            {"length": 1, "angle-limit": 0},
        ],
        "obstacle_config": [],
        "step_int": 8,
        "deg_step": 45,
    }

    oracle = CollisionOracle(test_setup)

    # Find two valid configs to use as start/goal
    valid_configs = []
    for i in range(45):
        for j in range(45):
            for k in range(45):
                cfg = (i, j, k)
                if oracle.is_valid(cfg):
                    valid_configs.append(cfg)
                if len(valid_configs) >= 2:
                    break
            if len(valid_configs) >= 2:
                break
        if len(valid_configs) >= 2:
            break

    # Pick start near one end, goal near the other, far apart
    start = valid_configs[0]
    # Find a config far from start
    goal = None
    for i in range(44, -1, -1):
        for j in range(44, -1, -1):
            for k in range(44, -1, -1):
                cfg = (i, j, k)
                if oracle.is_valid(cfg) and cfg != start:
                    if sum(abs(a - b) for a, b in zip(cfg, start)) > 20:
                        goal = cfg
                        break
            if goal:
                break
        if goal:
            break

    if goal is None:
        print("Could not find a suitable goal config for self-test.")
        sys.exit(1)

    print(f"Self-test: start={start}  goal={goal}")

    solver = LazyAStarSolver(test_setup, use_batched_oracle=True)
    path = solver.solve(start, goal)

    if path is None:
        print("No path found.")
        sys.exit(1)

    print(f"Path found: {len(path)} steps")

    # Verify axis-only moves
    for i in range(1, len(path)):
        diff = [abs(a - b) for a, b in zip(path[i], path[i - 1])]
        assert sum(diff) == 1 and max(diff) == 1, f"Step {i} is not axis-only: {diff}"

    # Verify every config is valid
    for i, cfg in enumerate(path):
        assert oracle.is_valid(cfg), f"Config at step {i} is invalid: {cfg}"

    print("All checks passed.")
    print("Stats:", solver.get_stats())

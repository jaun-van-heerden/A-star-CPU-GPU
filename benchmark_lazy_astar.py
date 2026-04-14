"""
benchmark_lazy_astar.py — Compare lazy A* against dense-grid precompute + A*.

Metrics collected per run
-------------------------
  wall-clock time   precompute phase, planning phase, total
  peak memory       via tracemalloc
  oracle calls      total geometry checks performed
  nodes expanded    A* nodes popped from the heap
  cache hit rate    checks answered from the memo cache
  path length       number of configs in the solution

Test cases
----------
  1. 3-joint coarse  (d=3, n=45)   — sanity, both approaches feasible
  2. 3-joint medium  (d=3, n=180)  — 5.8M cells, meaningful timing gap
  3. 4-joint medium  (d=4, n=45)   — 4.1M cells, still feasible to precompute
  4. 5-joint         (d=5, n=45)   — ~184M cells, precompute infeasible
  5. 6-joint         (d=6, n=45)   — ~8.3B cells, only lazy works
  6. 3-joint+obstacles               — verify obstacle avoidance
  7. Near vs far goal                — show how oracle calls scale with path length
"""

import time
import tracemalloc

import numpy as np

from lazy_astar import CollisionOracle, LazyAStarSolver
from aStarXd import AStarSolver
from calculate_cspace_seq import calculate_cspace as _seq_cspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_mib() -> float:
    snap = tracemalloc.take_snapshot()
    return sum(s.size for s in snap.statistics("lineno")) / 1024 / 1024


def find_valid_pair(setup, min_l1_distance: int = 10):
    """
    Scan the grid for two valid configs (start, goal) separated by at
    least min_l1_distance in L1 norm.  Returns (start, goal) or raises
    RuntimeError if none found within a reasonable search budget.
    """
    oracle = CollisionOracle(setup)
    n = setup["deg_step"]
    d = len(setup["arm_config"])

    found: list[tuple] = []
    # Walk a diagonal path through the grid to sample a spread of configs
    step = max(1, n // 12)
    for base in range(0, n, step):
        cfg = tuple(base for _ in range(d))
        if oracle.is_valid(cfg):
            if not found:
                found.append(cfg)
            else:
                if sum(abs(a - b) for a, b in zip(cfg, found[0])) >= min_l1_distance:
                    found.append(cfg)
                    return found[0], found[1]

    # Broader scan if diagonal walk failed
    for i in range(n):
        for j in range(n):
            cfg = (i, j) + tuple(n // 3 for _ in range(d - 2))
            if oracle.is_valid(cfg):
                if not found:
                    found.append(cfg)
                elif sum(abs(a - b) for a, b in zip(cfg, found[0])) >= min_l1_distance:
                    found.append(cfg)
                    return found[0], found[1]

    raise RuntimeError(
        f"Could not find two valid configs separated by L1 >= {min_l1_distance}. "
        "Try reducing min_l1_distance or adjusting setup."
    )


def verify_path(path, start, goal, setup):
    """Assert path endpoints, unit axis-only steps, and per-step oracle validity."""
    assert path[0] == start, f"Wrong start: {path[0]}"
    assert path[-1] == goal,  f"Wrong goal:  {path[-1]}"
    oracle = CollisionOracle(setup)
    for i in range(1, len(path)):
        diff = [abs(a - b) for a, b in zip(path[i], path[i - 1])]
        assert sum(diff) == 1 and max(diff) == 1, (
            f"Step {i-1}→{i} is not a unit axis-only move: diff={diff}"
        )
        assert oracle.is_valid(path[i]), f"Config at step {i} failed oracle: {path[i]}"
    return True


# ---------------------------------------------------------------------------
# Single benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    label: str,
    setup: dict,
    start: tuple,
    goal: tuple,
    skip_precomputed: bool = False,
):
    d    = len(setup["arm_config"])
    n    = setup["deg_step"]
    size = n ** d

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  d={d}  step_int={setup['step_int']}  deg_step={n}  "
          f"grid={size:,} cells")
    print(f"  start={start}  goal={goal}")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # Precomputed approach
    # ------------------------------------------------------------------
    if skip_precomputed:
        print("  [Precomputed]  SKIPPED — grid too large")
        pre_path = None
    else:
        try:
            tracemalloc.start()
            t0 = time.perf_counter()
            cspace = _seq_cspace(setup)
            t_pre  = time.perf_counter() - t0

            t1 = time.perf_counter()
            pre_solver = AStarSolver(cspace)
            pre_path   = pre_solver.solve(start, goal)
            t_plan_pre = time.perf_counter() - t1

            mem_pre = _peak_mib()
            tracemalloc.stop()

            print(
                f"  [Precomputed]  precompute={t_pre:.3f}s  "
                f"plan={t_plan_pre:.4f}s  "
                f"total={t_pre + t_plan_pre:.3f}s  "
                f"mem={mem_pre:.1f} MiB  "
                f"path={len(pre_path) if pre_path else None}"
            )
        except MemoryError:
            print("  [Precomputed]  SKIPPED — MemoryError")
            pre_path = None
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()

    # ------------------------------------------------------------------
    # Lazy A* — scalar oracle
    # ------------------------------------------------------------------
    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        solver_s = LazyAStarSolver(setup, use_batched_oracle=False)
        path_s   = solver_s.solve(start, goal)
        t_lazy_s = time.perf_counter() - t0
        mem_s    = _peak_mib()
        tracemalloc.stop()
        stats_s  = solver_s.get_stats()

        hit_rate_s = (
            stats_s["cache_hits"] / max(1, stats_s["cache_hits"] + stats_s["oracle_calls"])
        ) * 100

        print(
            f"  [Lazy scalar]  total={t_lazy_s:.4f}s  "
            f"mem={mem_s:.1f} MiB  "
            f"path={stats_s['path_length']}  "
            f"expanded={stats_s['nodes_expanded']}  "
            f"oracle={stats_s['oracle_calls']}  "
            f"cache_hit={hit_rate_s:.0f}%"
        )
    except ValueError as e:
        print(f"  [Lazy scalar]  SKIPPED — {e}")
        path_s = None
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    # ------------------------------------------------------------------
    # Lazy A* — batched oracle
    # ------------------------------------------------------------------
    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        solver_b = LazyAStarSolver(setup, use_batched_oracle=True)
        path_b   = solver_b.solve(start, goal)
        t_lazy_b = time.perf_counter() - t0
        mem_b    = _peak_mib()
        tracemalloc.stop()
        stats_b  = solver_b.get_stats()

        hit_rate_b = (
            stats_b["cache_hits"] / max(1, stats_b["cache_hits"] + stats_b["oracle_calls"])
        ) * 100

        print(
            f"  [Lazy batched] total={t_lazy_b:.4f}s  "
            f"mem={mem_b:.1f} MiB  "
            f"path={stats_b['path_length']}  "
            f"expanded={stats_b['nodes_expanded']}  "
            f"oracle={stats_b['oracle_calls']}  "
            f"cache_hit={hit_rate_b:.0f}%"
        )
    except ValueError as e:
        print(f"  [Lazy batched] SKIPPED — {e}")
        path_b = None
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    # ------------------------------------------------------------------
    # Verification (only when lazy found a path)
    # ------------------------------------------------------------------
    if path_b is not None:
        verify_path(path_b, start, goal, setup)
        print("  [Verify]       path OK (endpoints, axis-only steps, oracle-valid)")

    # ------------------------------------------------------------------
    # Speedup summary
    # ------------------------------------------------------------------
    if pre_path is not None and path_b is not None:
        total_pre  = t_pre + t_plan_pre
        speedup    = total_pre / max(t_lazy_b, 1e-9)
        pre_len    = len(pre_path)
        lazy_len   = stats_b["path_length"]
        print(
            f"  [Summary]      lazy is {speedup:.1f}x faster end-to-end  "
            f"(precomputed path={pre_len}, lazy path={lazy_len} — "
            f"different angle conventions)"
        )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def arm(n: int, length: float = 1.0, limit: float = 0.0) -> list:
    """Helper: build arm_config for n identical arms."""
    return [{"length": length, "angle-limit": limit} for _ in range(n)]


if __name__ == "__main__":
    print("Lazy A* vs Precomputed C-space benchmark")
    print("Note: precomputed uses cumulative-angle convention (calculate_cspace_seq);")
    print("      lazy uses local-relative convention (aStarRobotArm_no_amin.py).")
    print("      Path lengths may differ — timing/memory comparison is still valid.\n")

    # ------------------------------------------------------------------
    # Test 1: 3-joint coarse — sanity, both approaches run
    # ------------------------------------------------------------------
    s1 = {"arm_config": arm(3), "obstacle_config": [], "step_int": 8, "deg_step": 45}
    start1, goal1 = find_valid_pair(s1, min_l1_distance=20)
    run_benchmark("Test 1 — 3-joint coarse (d=3, n=45) sanity", s1, start1, goal1)

    # ------------------------------------------------------------------
    # Test 2: 3-joint medium
    # ------------------------------------------------------------------
    s2 = {"arm_config": arm(3), "obstacle_config": [], "step_int": 2, "deg_step": 180}
    start2, goal2 = find_valid_pair(s2, min_l1_distance=60)
    run_benchmark("Test 2 — 3-joint medium (d=3, n=180)", s2, start2, goal2)

    # ------------------------------------------------------------------
    # Test 3: 4-joint medium
    # ------------------------------------------------------------------
    s3 = {"arm_config": arm(4), "obstacle_config": [], "step_int": 8, "deg_step": 45}
    start3, goal3 = find_valid_pair(s3, min_l1_distance=20)
    run_benchmark("Test 3 — 4-joint medium (d=4, n=45)", s3, start3, goal3)

    # ------------------------------------------------------------------
    # Test 4: 5-joint  (precompute infeasible, ~184M cells, >1 GB)
    # ------------------------------------------------------------------
    s4 = {"arm_config": arm(5), "obstacle_config": [], "step_int": 8, "deg_step": 45}
    start4, goal4 = find_valid_pair(s4, min_l1_distance=20)
    run_benchmark(
        "Test 4 — 5-joint (d=5, n=45) precompute infeasible",
        s4, start4, goal4,
        skip_precomputed=True,
    )

    # ------------------------------------------------------------------
    # Test 5: 6-joint  (only lazy works)
    # ------------------------------------------------------------------
    s5 = {"arm_config": arm(6), "obstacle_config": [], "step_int": 8, "deg_step": 45}
    start5, goal5 = find_valid_pair(s5, min_l1_distance=20)
    run_benchmark(
        "Test 5 — 6-joint (d=6, n=45) only lazy works",
        s5, start5, goal5,
        skip_precomputed=True,
    )

    # ------------------------------------------------------------------
    # Test 6: 3-joint with obstacles
    # ------------------------------------------------------------------
    s6 = {
        "arm_config": arm(3),
        "obstacle_config": [
            (complex(-2, 1), complex(-2, 0)),
            (complex(-1, -2), complex(-1, -2)),
            (complex(1, 1),  complex(3,  1)),
        ],
        "step_int": 8,
        "deg_step": 45,
    }
    start6, goal6 = find_valid_pair(s6, min_l1_distance=15)
    run_benchmark("Test 6 — 3-joint with obstacles (d=3, n=45)", s6, start6, goal6)

    # ------------------------------------------------------------------
    # Test 7a/b: Near vs far goal — show oracle call scaling
    # ------------------------------------------------------------------
    s7 = {"arm_config": arm(3), "obstacle_config": [], "step_int": 8, "deg_step": 45}
    near_start, near_goal = find_valid_pair(s7, min_l1_distance=5)
    far_start,  far_goal  = find_valid_pair(s7, min_l1_distance=40)

    run_benchmark("Test 7a — 3-joint near goal", s7, near_start, near_goal)
    run_benchmark("Test 7b — 3-joint far  goal", s7, far_start,  far_goal)

    print("\nBenchmark complete.")

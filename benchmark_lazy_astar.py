"""
benchmark_lazy_astar.py — Lazy A* scaling benchmark.

Sections
--------
  A. Scaling with joint count  d = 3 … 10  (open workspace)
  B. Obstacle complexity       d = 3, progressively harder maps
  C. Scalar vs batched oracle  crossover analysis

Note: precomputed approach (calculate_cspace_seq) uses the cumulative-absolute
angle convention; lazy A* uses the local-relative convention from
aStarRobotArm_no_amin.py.  They produce different C-spaces, so path lengths
will not match — the timing/memory/scalability comparison is still valid.
Precomputed is skipped automatically when the grid exceeds SKIP_THRESHOLD.
"""

import time
import tracemalloc

import numpy as np

from lazy_astar import CollisionOracle, LazyAStarSolver

# Cells above this threshold → skip the precomputed approach automatically
SKIP_THRESHOLD = 100_000


# ---------------------------------------------------------------------------
# Obstacle library
# ---------------------------------------------------------------------------

OBSTACLES = {
    "none": [],

    "light": [
        # Two isolated bars in different quadrants
        (complex(0.8,  1.2), complex(1.8,  1.2)),
        (complex(-1.5, -0.8), complex(-0.6, -0.8)),
    ],

    "dense": [
        # Six segments cluttering the workspace
        (complex(0.5,  1.0), complex(1.5,  1.0)),
        (complex(-1.5, 1.0), complex(-0.5, 1.0)),
        (complex(1.5, -1.0), complex(2.5, -1.0)),
        (complex(-2.0, 0.5), complex(-1.0, 0.5)),
        (complex(-2.0,-0.5), complex(-1.0,-0.5)),
        (complex(0.0,  1.8), complex(0.0,  2.5)),
    ],

    "wall_gap": [
        # Near-complete horizontal wall at y=0.8 with a narrow gap at x=0
        (complex(-3.0,  0.8), complex(-0.25,  0.8)),
        (complex( 0.25,  0.8), complex( 3.0,  0.8)),
        # Mirror wall at y=-0.8
        (complex(-3.0, -0.8), complex(-0.25, -0.8)),
        (complex( 0.25, -0.8), complex( 3.0, -0.8)),
    ],

    "maze": [
        # Interlocking segments that create a maze-like workspace
        (complex(-2.0,  1.5), complex( 0.0,  1.5)),   # top-left shelf
        (complex( 0.0,  1.5), complex( 0.0,  0.5)),   # vertical drop
        (complex( 0.0,  0.5), complex( 2.0,  0.5)),   # mid-right shelf
        (complex(-2.0, -0.5), complex( 0.0, -0.5)),   # mid-left shelf
        (complex( 0.0, -0.5), complex( 0.0, -1.5)),   # vertical drop
        (complex( 0.0, -1.5), complex( 2.0, -1.5)),   # bottom-right shelf
    ],
}


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def find_valid_pair(setup, min_l1: int = 10, budget: int = 8000, seed: int = 42):
    """
    Find two valid configs separated by at least min_l1 L1-distance.
    Uses a diagonal scan first, then random sampling.
    Returns (start, goal) or raises RuntimeError.
    """
    oracle = CollisionOracle(setup)
    n = setup["deg_step"]
    d = len(setup["arm_config"])
    rng = np.random.default_rng(seed)
    found: list[tuple] = []

    def accept(cfg):
        if not found:
            if oracle.is_valid(cfg):
                found.append(cfg)
        elif sum(abs(a - b) for a, b in zip(cfg, found[0])) >= min_l1:
            if oracle.is_valid(cfg):
                return True
        return False

    # 1. Diagonal scan — fast, covers a wide spread
    step = max(1, n // 14)
    for base in range(0, n, step):
        if accept(tuple(base for _ in range(d))):
            return found[0], tuple(base for _ in range(d))

    # 2. Random search — good for high-d where the diagonal is sparse
    for _ in range(budget):
        cfg = tuple(int(x) for x in rng.integers(0, n, size=d))
        if accept(cfg):
            return found[0], cfg

    if len(found) >= 1:
        # Last resort: return same config twice so benchmark can at least run
        raise RuntimeError(
            f"Only one valid config found (d={d}, n={n}, L1>={min_l1}). "
            "Increase budget or reduce min_l1."
        )
    raise RuntimeError(f"No valid configs found (d={d}, n={n}).")


# ---------------------------------------------------------------------------
# Path verification
# ---------------------------------------------------------------------------

def verify_path(path, start, goal, setup):
    """Assert endpoints, unit axis-only steps, and per-step oracle validity."""
    oracle = CollisionOracle(setup)
    assert path[0] == start, f"Wrong start: {path[0]}"
    assert path[-1] == goal,  f"Wrong goal:  {path[-1]}"
    for i in range(1, len(path)):
        diff = [abs(a - b) for a, b in zip(path[i], path[i - 1])]
        assert sum(diff) == 1 and max(diff) == 1, \
            f"Non-unit axis-only step at {i}: diff={diff}"
        assert oracle.is_valid(path[i]), f"Invalid config at step {i}: {path[i]}"


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def _peak_mib() -> float:
    snap = tracemalloc.take_snapshot()
    return sum(s.size for s in snap.statistics("lineno")) / 1024 / 1024


def run_one(label, setup, start, goal, run_precomputed=True):
    from aStarXd import AStarSolver
    from calculate_cspace_seq import calculate_cspace as seq_cspace

    d    = len(setup["arm_config"])
    n    = setup["deg_step"]
    size = n ** d
    obs  = len(setup.get("obstacle_config", []))

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  d={d}  step={setup['step_int']}°/idx  n={n}  "
          f"grid={size:>15,}  obstacles={obs}")
    print(f"  start={start}")
    print(f"  goal ={goal}  (L1={sum(abs(a-b) for a,b in zip(start,goal))})")
    print(f"{'='*70}")

    results = {}

    # ------------------------------------------------------------------
    # Precomputed  (skip if grid too big)
    # ------------------------------------------------------------------
    auto_skip = size > SKIP_THRESHOLD
    if not run_precomputed or auto_skip:
        if auto_skip:
            print(f"  [Precomputed]  SKIP — {size:,} cells > {SKIP_THRESHOLD:,} limit")
        else:
            print(f"  [Precomputed]  SKIP")
    else:
        try:
            tracemalloc.start()
            t0     = time.perf_counter()
            cspace = seq_cspace(setup)
            t_pre  = time.perf_counter() - t0
            t1     = time.perf_counter()
            ppath  = AStarSolver(cspace).solve(start, goal)
            t_plan = time.perf_counter() - t1
            mem    = _peak_mib()
            tracemalloc.stop()
            plen = len(ppath) if ppath else "None*"
            print(f"  [Precomputed]  precompute={t_pre:.3f}s  plan={t_plan:.4f}s  "
                  f"total={t_pre+t_plan:.3f}s  mem={mem:.1f}MiB  path={plen}")
            if not ppath:
                print("                 *None expected — different angle convention")
            results["precomputed"] = (ppath, t_pre + t_plan)
        except MemoryError:
            print("  [Precomputed]  SKIP — MemoryError")
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()

    # ------------------------------------------------------------------
    # Lazy A*  (scalar and batched)
    # ------------------------------------------------------------------
    for mode, batched in [("scalar ", False), ("batched", True)]:
        try:
            tracemalloc.start()
            t0     = time.perf_counter()
            solver = LazyAStarSolver(setup, use_batched_oracle=batched)
            path   = solver.solve(start, goal)
            t      = time.perf_counter() - t0
            mem    = _peak_mib()
            tracemalloc.stop()
            st     = solver.get_stats()
            total  = st["oracle_calls"] + st["cache_hits"]
            hit    = 100 * st["cache_hits"] / max(1, total)
            status = f"path={st['path_length']}" if path else "NO PATH"
            print(f"  [Lazy {mode}]  {t:.4f}s  mem={mem:.1f}MiB  {status}  "
                  f"expanded={st['nodes_expanded']}  "
                  f"oracle={st['oracle_calls']}  cache={hit:.0f}%")
            results[mode.strip()] = (path, t, st)
        except ValueError as e:
            print(f"  [Lazy {mode}]  SKIP — {e}")
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()

    # Verify batched path
    bpath = results.get("batched", (None,))[0]
    if bpath is not None:
        verify_path(bpath, start, goal, setup)
        print("  [Verify]       OK")

    # Speedup: scalar vs batched
    sr = results.get("scalar")
    br = results.get("batched")
    if sr and br and len(sr) > 1 and len(br) > 1:
        ts, tb = sr[1], br[1]
        faster = "batched" if tb < ts else "scalar "
        ratio  = max(ts, tb) / max(min(ts, tb), 1e-9)
        print(f"  [Mode delta]   {faster} is {ratio:.1f}x faster")

    return results


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

def arm(d, length=1.0, limit=0.0):
    return [{"length": length, "angle-limit": limit} for _ in range(d)]


def section(title):
    print(f"\n\n{'#'*70}")
    print(f"#  {title}")
    print(f"{'#'*70}")


if __name__ == "__main__":

    print("Lazy A* Scalability Benchmark")
    print("="*70)

    # ======================================================================
    # SECTION A — Scaling with joint count (open workspace, no obstacles)
    # ======================================================================
    section("A. Scaling with joint count — open workspace")

    joint_results = {}   # d → (time_scalar, time_batched, nodes_expanded, oracle_calls)

    for d, step, min_l1 in [
        (3,  8, 20),
        (4,  8, 20),
        (5,  8, 15),
        (6,  8, 12),
        (7,  8, 10),
        (8,  8,  8),
        (10, 8,  6),
    ]:
        setup = {
            "arm_config":     arm(d),
            "obstacle_config": [],
            "step_int":        step,
            "deg_step":        360 // step,
        }
        try:
            start, goal = find_valid_pair(setup, min_l1=min_l1, budget=10_000)
        except RuntimeError as e:
            print(f"\n  d={d}: SKIP — {e}")
            continue

        res = run_one(f"A{d}. {d}-joint open space", setup, start, goal)
        sr  = res.get("scalar")
        br  = res.get("batched")
        if sr and br and len(sr) > 1:
            joint_results[d] = (sr[1], br[1],
                                br[2]["nodes_expanded"] if br[2] else None,
                                br[2]["oracle_calls"]   if br[2] else None)

    # Summary table
    if joint_results:
        print("\n\n  Joint-count scaling summary (batched oracle)")
        print(f"  {'d':>3}  {'time(s)':>8}  {'nodes':>8}  {'oracle':>8}")
        print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}")
        for d, (ts, tb, nodes, oracle) in sorted(joint_results.items()):
            print(f"  {d:>3}  {tb:>8.4f}  {nodes if nodes is not None else '?':>8}  "
                  f"{oracle if oracle is not None else '?':>8}")

    # ======================================================================
    # SECTION B — Obstacle complexity (d=3, fixed step)
    # ======================================================================
    section("B. Obstacle complexity — d=3, step=8°")

    obstacle_results = {}

    for obs_name, obs_segs in OBSTACLES.items():
        setup = {
            "arm_config":     arm(3),
            "obstacle_config": obs_segs,
            "step_int":        8,
            "deg_step":        45,
        }
        try:
            start, goal = find_valid_pair(setup, min_l1=15, budget=5000)
        except RuntimeError as e:
            print(f"\n  {obs_name}: SKIP — {e}")
            continue

        res = run_one(
            f"B. d=3, obstacles={obs_name} ({len(obs_segs)} segs)",
            setup, start, goal,
            run_precomputed=(obs_name == "none"),   # only compare precomputed once
        )
        br = res.get("batched")
        if br and len(br) > 1:
            obstacle_results[obs_name] = (br[1], br[2])

    # Summary table
    if obstacle_results:
        print("\n\n  Obstacle complexity summary (batched oracle, d=3)")
        print(f"  {'map':>10}  {'segs':>5}  {'time(s)':>8}  {'nodes':>8}  "
              f"{'oracle':>8}  {'path':>6}")
        print(f"  {'-'*10}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
        for obs_name, (t, st) in obstacle_results.items():
            segs  = len(OBSTACLES[obs_name])
            nodes = st["nodes_expanded"] if st else "?"
            oracle= st["oracle_calls"]   if st else "?"
            plen  = st["path_length"]    if st else "?"
            print(f"  {obs_name:>10}  {segs:>5}  {t:>8.4f}  {nodes:>8}  "
                  f"{oracle:>8}  {plen:>6}")

    # ======================================================================
    # SECTION C — High-d with dense obstacles
    # ======================================================================
    section("C. High-d arms with dense obstacles")

    for d, min_l1 in [(4, 12), (5, 8), (6, 6)]:
        setup = {
            "arm_config":     arm(d),
            "obstacle_config": OBSTACLES["dense"],
            "step_int":        8,
            "deg_step":        45,
        }
        try:
            start, goal = find_valid_pair(setup, min_l1=min_l1, budget=10_000)
        except RuntimeError as e:
            print(f"\n  d={d} dense: SKIP — {e}")
            continue
        run_one(f"C{d}. {d}-joint dense obstacles", setup, start, goal)

    print("\n\nBenchmark complete.")

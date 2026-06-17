import heapq
from itertools import product
from typing import Callable, Optional


class LazyAStarSolver:
    """
    Domain-agnostic A* over an N-dimensional discrete joint-space.
    Passability is determined lazily via a caller-supplied valid_fn —
    no precomputed grid needed. Scales to high-dimensional spaces
    (e.g. many-link robot arms) where precomputing D^N cells is infeasible.
    Heuristic: Chebyshev (L∞) — admissible for unit-cost diagonal moves.
    """

    def __init__(self, ndim: int, valid_fn: Callable[[tuple], bool], deg_step: int):
        """
        ndim      — number of discrete axes (joints)
        valid_fn  — returns True if a position tuple is collision-free and in-bounds
        deg_step  — number of discrete steps per axis (used only for neighbour wrapping guard)
        """
        self.ndim = ndim
        self.valid_fn = valid_fn
        self.deg_step = deg_step
        # Precompute all 3^N - 1 non-zero neighbour offsets once
        self._offsets = [
            o for o in product([-1, 0, 1], repeat=ndim)
            if any(x != 0 for x in o)
        ]

    def _heuristic(self, a: tuple, b: tuple) -> int:
        return max(abs(x - y) for x, y in zip(a, b))

    def _neighbours(self, pos: tuple):
        for offset in self._offsets:
            nb = tuple(p + o for p, o in zip(pos, offset))
            if self.valid_fn(nb):
                yield nb

    def solve(
        self,
        start: tuple,
        goal: tuple,
        max_nodes: int = 500_000,
        weight: float = 1.0,
    ) -> Optional[list]:
        """Return a path [start, ..., goal] or None if unreachable/over budget.

        weight > 1 enables weighted A* (WA*): paths are within `weight` × optimal
        length but the search volume shrinks by roughly weight^N, making it
        practical for high-dimensional spaces (many joints).
        """
        if not self.valid_fn(start) or not self.valid_fn(goal):
            return None

        h0 = self._heuristic(start, goal)
        open_list = [(weight * h0, start)]
        came_from: dict = {}
        best_cost: dict = {start: 0}
        visited: set = set()
        nodes = 0

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            if current in visited:
                continue
            visited.add(current)
            nodes += 1
            if nodes > max_nodes:
                return None

            g = best_cost[current]
            for nb in self._neighbours(current):
                next_cost = g + 1
                if next_cost < best_cost.get(nb, float('inf')):
                    best_cost[nb] = next_cost
                    came_from[nb] = current
                    heapq.heappush(
                        open_list,
                        (next_cost + weight * self._heuristic(nb, goal), nb),
                    )

        return None

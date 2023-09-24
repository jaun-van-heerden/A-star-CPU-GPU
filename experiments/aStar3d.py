import numpy as np
import heapq

class Node:
    def __init__(self, position, cost=0, heuristic=0):
        self.position = position
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost


class AStarSolver:
    def __init__(self, grid):
        self.grid = grid

    def _in_bounds(self, position):
        return all(0 <= idx < dim for idx, dim in zip(position, self.grid.shape))

    def _passable(self, position):
        return self.grid[position] == 1

    def _neighbors(self, position):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    # Skip the current position (no movement)
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    next_position = (position[0] + dx, position[1] + dy, position[2] + dz)
                    
                    if self._in_bounds(next_position) and self._passable(next_position):
                        yield next_position


    def _heuristic(self, current, goal):
        return sum((a - b) ** 2 for a, b in zip(current, goal)) ** 0.5

    def solve(self, start, goal):
        open_list = [Node(start, heuristic=self._heuristic(start, goal))]
        came_from = {}
        visited = set()

        while open_list:
            current_node = heapq.heappop(open_list)

            if current_node.position == goal:
                path = [current_node.position]
                while current_node.position in came_from:
                    current_node = came_from[current_node.position]
                    path.append(current_node.position)
                return path[::-1]

            visited.add(current_node.position)

            for neighbor in self._neighbors(current_node.position):
                if neighbor in visited:
                    continue
                next_cost = current_node.cost + 1
                next_node = Node(neighbor, next_cost, self._heuristic(neighbor, goal))
                if neighbor not in [node.position for node in open_list] or next_cost < current_node.cost:
                    came_from[neighbor] = current_node
                    heapq.heappush(open_list, next_node)

        return None  # No path found


if __name__ == '__main__':
    grid = np.array([
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1]
        ],
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 1]
        ]
    ])

    solver = AStarSolver(grid)
    start, goal = (0, 0, 0), (2, 2, 2)
    path = solver.solve(start, goal)

    if path:
        print(f"Path from {start} to {goal}:")
        for step in path:
            print(step)
    else:
        print(f"No path found from {start} to {goal}.")

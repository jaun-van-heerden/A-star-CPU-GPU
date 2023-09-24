from flask import Flask, render_template, jsonify, request

from concurrent.futures import ThreadPoolExecutor

from multiprocessing import Pool, cpu_count
import time


import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = Flask(__name__)


class Node:
    def __init__(self, x, y, walkable=True):
        self.x = x
        self.y = y
        self.walkable = walkable
        self.parent = None
        self.g = float('inf')  # distance from starting node
        self.h = 0  # heuristic: estimated distance to the end node
        self.f = 0  # total cost: g + h

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def manhattan_distance(node1, node2):
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)

def compute_heuristics(data):
    neighbor, end = data
    return manhattan_distance(neighbor, end)

def a_star(grid, start, end, parallel=False):
    open_list = [start]
    closed_list = []

    start.g = 0
    start.h = manhattan_distance(start, end)
    start.f = start.g + start.h

    while open_list:
        current_node = min(open_list, key=lambda node: node.f)
        open_list.remove(current_node)
        closed_list.append(current_node)

        if current_node == end:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        neighbors = []
        for x, y in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = current_node.x + x, current_node.y + y
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]):
                neighbors.append(grid[new_x][new_y])

        if parallel:
            with Pool(cpu_count()) as pool:
                heuristics = pool.map(compute_heuristics, [(neighbor, end) for neighbor in neighbors])
            for neighbor, h in zip(neighbors, heuristics):
                neighbor.h = h
        else:
            for neighbor in neighbors:
                neighbor.h = manhattan_distance(neighbor, end)

        for neighbor in neighbors:
            if neighbor not in closed_list and neighbor.walkable:
                if neighbor in open_list:
                    new_g = current_node.g + 1
                    if new_g < neighbor.g:
                        neighbor.g = new_g
                        neighbor.parent = current_node
                else:
                    neighbor.g = current_node.g + 1
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current_node
                    open_list.append(neighbor)

    return None  # No path found






def visualize(grid, path=None):
    fig, ax = plt.subplots()

    for row in grid:
        for node in row:
            color = 'white'  # default is walkable
            if not node.walkable:
                color = 'black'  # non-walkable
            elif node == start_node:
                color = 'green'  # start
            elif node == end_node:
                color = 'red'    # end

            ax.add_patch(patches.Rectangle((node.x, node.y), 1, 1, facecolor=color))

    if path:
        for x, y in path[1:-1]:  # not visualizing start/end nodes here
            ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='blue'))

    plt.xlim(0, len(grid))
    plt.ylim(0, len(grid[0]))
    plt.gca().invert_yaxis()  # so (0,0) is top-left corner
    plt.show()




# Example usage:
grid = [[Node(i, j) for j in range(5)] for i in range(5)]
grid[2][1].walkable = False
grid[2][2].walkable = False
grid[2][3].walkable = False
start_node = grid[0][0]
end_node = grid[4][4]

path = a_star(grid, start_node, end_node)
#visualize(grid, path)





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_path', methods=['POST'])
def find_path():
    data = request.json
    grid = [[Node(i, j, not cell['isWall']) for j, cell in enumerate(row)] for i, row in enumerate(data['grid'])]
    start_node = grid[data['start']['x']][data['start']['y']]
    end_node = grid[data['end']['x']][data['end']['y']]
    print('finding path...')
    start_time = time.time()
    path = a_star(grid, start_node, end_node, parallel=False)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return jsonify(path)

@app.route('/find_path_parallel', methods=['POST'])
def find_path_parallel():
    data = request.json
    grid = [[Node(i, j, not cell['isWall']) for j, cell in enumerate(row)] for i, row in enumerate(data['grid'])]
    start_node = grid[data['start']['x']][data['start']['y']]
    end_node = grid[data['end']['x']][data['end']['y']]
    print('finding path parallel...')
    start_time = time.time()
    path = a_star(grid, start_node, end_node, parallel=True)
    end_time = time.time()
    print(f"Time taken (parallel): {end_time - start_time} seconds")
    return jsonify(path)


if __name__ == '__main__':
    app.run()
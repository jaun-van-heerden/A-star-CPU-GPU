import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque, defaultdict

class Node:
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []
        self.level = 0
        self.x = 0

    def connect_to(self, node):
        self.outputs.append(node)
        node.inputs.append(self)

def barycenter_horizontal_placement(node, levels):
    if node.outputs:
        node.x = sum(out_node.x for out_node in node.outputs) / len(node.outputs)
        levels[node.level].sort(key=lambda n: n.x)

def auto_layout_coordinates(nodes, rect_width, rect_height):
    visited = set()
    queue = [node for node in nodes if not node.inputs]

    for node in queue:
        visited.add(node)

    while queue:
        current_node = queue.pop(0)
        for output_node in current_node.outputs:
            if output_node not in visited:
                output_node.level = current_node.level + 1
                visited.add(output_node)
                queue.append(output_node)

    levels = defaultdict(list)
    for node in nodes:
        levels[node.level].append(node)
        node.x = len(levels[node.level]) - 1

    for _ in range(5):
        for node in nodes:
            barycenter_horizontal_placement(node, levels)

    coordinates = {}
    for level, nodes_on_level in levels.items():
        for node in nodes_on_level:
            coordinates[node.name] = (level * rect_width, node.x * rect_height)

    return coordinates

def visualize_layout(nodes):
    rect_width, rect_height = 0.7, 0.6
    coords = auto_layout_coordinates(nodes, rect_width, rect_height)
    
    fig, ax = plt.subplots()
    for node in nodes:
        x, y = coords[node.name]
        ax.add_patch(patches.Rectangle((x, y), rect_width, rect_height, color='skyblue'))
        ax.text(x + rect_width / 2, y + rect_height / 2, node.name, ha='center', va='center')
        
        for output_node in node.outputs:
            x2, y2 = coords[output_node.name]
            ax.arrow(x + rect_width, y + rect_height / 2, x2 - x - rect_width, y2 - y, head_width=0.1, head_length=0.1)

    ax.axis('off')
    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.show()

if __name__ == "__main__":
    nodes = [Node(letter) for letter in ["A", "B", "C", "D", "E", "F", "G", "H", "X"]]
    A, B, C, D, E, F, G, H, X = nodes

    A.connect_to(B)
    A.connect_to(C)
    B.connect_to(D)
    B.connect_to(E)
    C.connect_to(F)
    C.connect_to(G)
    F.connect_to(H)
    X.connect_to(B)

    visualize_layout(nodes)

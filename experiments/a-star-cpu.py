from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
import heapq



def color_difference(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))



def heuristic(node, end):
    return color_difference(image_array[node], image_array[end])

def cost_to_move(color1, target_color):
    return 255 - color_difference(color1, target_color)



# def heuristic(node, end):
#     return np.linalg.norm(image_array[node] - image_array[end])


def is_valid_move(neighbor: Tuple[int, int], matrix: List[List[int]]) -> bool:
    return 0 <= neighbor[0] < len(matrix) and 0 <= neighbor[1] < len(matrix[0]) and matrix[neighbor[0]][neighbor[1]] != 1

def reconstruct_path(came_from: Dict[Tuple[int, int], Tuple[int, int]], current_node: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    while current_node in came_from:
        path.insert(0, current_node)
        current_node = came_from[current_node]
    return path



def astar(image_array, start, end, target_color):

    
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]
    open_list = [(0, start)]
    came_from = {}
    g_score = {node: float('inf') for node in np.ndindex(image_array.shape[:2])}
    g_score[start] = 0

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)
        
        if current_node == end:
            # Reconstruct the path
            path = []
            while current_node in came_from:
                path.insert(0, current_node)
                current_node = came_from[current_node]
            return path

        for move in moves:
            neighbor = tuple(map(sum, zip(current_node, move)))
            if 0 <= neighbor[0] < image_array.shape[0] and 0 <= neighbor[1] < image_array.shape[1]:
                tentative_g_score = g_score[current_node] + cost_to_move(image_array[neighbor], target_color) #+ image_array[neighbor]

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score, neighbor))

    return None


def draw_path_on_image(image_array, path, color):
    for point in path:
        cv2.circle(image_array, (point[1], point[0]), 1, color, -1)

if __name__ == "__main__":
    #image_path = 'ferrari_02.jpg'
    image_path = 'track_02.jpg'
    image = Image.open(image_path)
    image_array = np.array(image)
    
    print(image_array.shape[0]//2)

    start = (image_array.shape[0]//2, 0)
    end = (image_array.shape[0]//2, image_array.shape[1] - 1)

    # Find the shortest paths for R, G, and B
    
    target_color = (150, 150, 150) 
    
    shortest_path = astar(image_array, start, end, target_color)
    #shortest_path_R = astar(image_array[:,:,0], start, end)
    # shortest_path_G = astar(image_array[:,:,1], start, end)
    # shortest_path_B = astar(image_array[:,:,2], start, end)

    # Draw the paths on the image
    draw_path_on_image(image_array, shortest_path, [255, 0, 0])  # Red
    # draw_path_on_image(image_array, shortest_path_G, [0, 255, 0])  # Green
    # draw_path_on_image(image_array, shortest_path_B, [0, 0, 255])  # Blue

    # Display the image with the paths
    plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    plt.title('Image with R, G, B Paths')
    plt.show()




import numpy as np
import matplotlib.pyplot as plt
import heapq
from matplotlib.animation import FuncAnimation

# Robotic Arm Configuration
L1 = 1  # length of first link
L2 = 1  # length of second link



def calculate_valid_space(c_space):
    
    
    
    
    
    
    
    


# Define the minimum angle and the obstacle
MIN_ANGLE = 30  # in degrees
OBSTACLE_CENTER = (0.5, 0.5)
OBSTACLE_RADIUS = 0.25

# Improved collision detection
def check_collision(theta1, theta2):
    x1, y1 = L1 * np.cos(np.radians(theta1)), L1 * np.sin(np.radians(theta1))
    x2, y2 = x1 + L2 * np.cos(np.radians(theta1 + theta2)), y1 + L2 * np.sin(np.radians(theta1 + theta2))

    # Discretize the links and check collision for each segment
    for fraction in np.linspace(0, 1, 10):  # 10 segments for each link
        x_link1 = fraction * x1
        y_link1 = fraction * y1
        
        x_link2 = x1 + fraction * (x2 - x1)
        y_link2 = y1 + fraction * (y2 - y1)
        
        for x, y in [(x_link1, y_link1), (x_link2, y_link2)]:
            if (x - OBSTACLE_CENTER[0])**2 + (y - OBSTACLE_CENTER[1])**2 <= OBSTACLE_RADIUS**2:
                return True

    return False


# Identify invalid configurations
invalid_configs = []

for theta1 in range(361):
    for theta2 in range(361):
        # Check for minimum angle
        if abs(theta2) < MIN_ANGLE:
            invalid_configs.append((theta1, theta2))
        # Check for collisions
        elif check_collision(theta1, theta2):
            invalid_configs.append((theta1, theta2))

def forward_kinematics(theta1, theta2):
    x1 = L1 * np.cos(np.radians(theta1))
    y1 = L1 * np.sin(np.radians(theta1))
    x2 = x1 + L2 * np.cos(np.radians(theta1 + theta2))
    y2 = y1 + L2 * np.sin(np.radians(theta1 + theta2))
    return x2, y2

def heuristic(node, end):
    x1, y1 = forward_kinematics(*node)
    x2, y2 = forward_kinematics(*end)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def is_valid_move(neighbor):
    return 0 <= neighbor[0] <= 360 and 0 <= neighbor[1] <= 360

def astar(start, end):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    open_list = [(0, start)]
    came_from = {}
    g_score = {node: float('inf') for node in [(i, j) for i in range(361) for j in range(361)]}
    g_score[start] = 0
    
    while open_list:
        current_cost, current_node = heapq.heappop(open_list)
        
        if current_node == end:
            path = []
            while current_node in came_from:
                path.insert(0, current_node)
                current_node = came_from[current_node]
            return path

        for move in moves:
            neighbor = (current_node[0] + move[0], current_node[1] + move[1])
            if is_valid_move(neighbor):
                tentative_g_score = g_score[current_node] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score, neighbor))
    return None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Setting the axes limits
ax1.set_xlim(-L1 - L2, L1 + L2)
ax1.set_ylim(-L1 - L2, L1 + L2)
ax2.set_xlim(0, 360)
ax2.set_ylim(0, 360)

line1, = ax1.plot([], [], '-o', mfc='red')
point, = ax2.plot([], [], 'ro')

def init():
    line1.set_data([], [])
    point.set_data([], [])
    return line1, point

def animate(config):
    # Robotic arm animation
    points = [(0, 0)]
    theta1, theta2 = config
    x1 = L1 * np.cos(np.radians(theta1))
    y1 = L1 * np.sin(np.radians(theta1))
    points.append((x1, y1))
    x2 = x1 + L2 * np.cos(np.radians(theta1 + theta2))
    y2 = y1 + L2 * np.sin(np.radians(theta1 + theta2))
    points.append((x2, y2))
    xs, ys = zip(*points)
    line1.set_data(xs, ys)

    # C-space animation
    point.set_data(theta1, theta2)
    
    return line1, point

if __name__ == "__main__":
    
    # create c-space grid
    c_space = np.zeros((361, 361))
    
    arm_config = {
        'arms': [{'name': 'arm01', 'length': 1, 'angle-limit':{'min': 10, 'max': 100}},
                 {'name': 'arm02', 'length': 2, 'angle-limit':{'min': 30, 'max': 100}},
                 {'name': 'arm03', 'length': 1, 'angle-limit':{'min': 30, 'max': 100}}]
        }
    
    # calculate valid areas
    c_space_valid = calculate_valid_space(c_space, arm_config)
    
    # find shortest path between two configurations
    
    # # Plot the invalid configurations on c-space
    # invalid_xs, invalid_ys = zip(*invalid_configs)
    # ax2.scatter(invalid_xs, invalid_ys, c='grey', marker='.', s=1)
    
    
    # start_config = (100, 30)
    # end_config = (180, 180)
    # path = astar(start_config, end_config)



    # ani = FuncAnimation(fig, animate, frames=path, init_func=init, blit=True, interval=50)
    # plt.tight_layout()
    # plt.show()

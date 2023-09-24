import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from aStar3d import AStarSolver


DEG_INT = 10

DEG_STEP = 360//DEG_INT


def animate_solution(solution, arm_config, x):
    fig, ax = plt.subplots()
    
    total_len = sum([arm['length'] for arm in arm_config])
    ax.set_xlim(-total_len, total_len)
    ax.set_ylim(-total_len, total_len)
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    
    line, = ax.plot([], [], 'o-')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        config = solution[frame]
        total_angle = 0
        xdata, ydata = [0], [0]
        for angle, arm in zip(config, arm_config):
            total_angle += DEG_INT * angle * np.pi / 180
            end_x, end_y = xdata[-1] + arm['length'] * np.cos(total_angle), ydata[-1] + arm['length'] * np.sin(total_angle)
            xdata.append(end_x)
            ydata.append(end_y)
        line.set_data(xdata, ydata)
        return line,

    ani = FuncAnimation(fig, update, frames=len(solution), init_func=init, blit=True, repeat=False)
    plt.show()



def select_random_config(c_space, val):
    indices = np.argwhere(c_space == val)
    if len(indices) == 0:
        return None
    random_idx = indices[np.random.choice(len(indices))]
    return tuple(random_idx)


def plot_arm_configuration(config, arm_config):
    fig, ax = plt.subplots()
    total_angle = 0
    start_point = complex(0, 0)
    for angle, arm in zip(config, arm_config):
        total_angle += DEG_INT * angle * np.pi / 180
        end_point = start_point + arm['length'] * np.exp(1j * total_angle)
        ax.plot([start_point.real, end_point.real], [start_point.imag, end_point.imag], 'o-')
        start_point = end_point
    ax.set_xlim(-sum([arm['length'] for arm in arm_config]), sum([arm['length'] for arm in arm_config]))
    ax.set_ylim(-sum([arm['length'] for arm in arm_config]), sum([arm['length'] for arm in arm_config]))
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.show()





def segment_intersection(p1, p2, q1, q2):
    """ Check if segments (p1, p2) and (q1, q2) intersect """
    def ccw(A, B, C):
        """ Check if points are listed in counterclockwise order """
        return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)

    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


def validate(config, arm_config):
    total_angle = 0
    positions = [complex(0, 0)]
    
    # check that there is no self-intersection
    for angle, arm in zip(config, arm_config):
        #print(DEG_INT * angle, 'degrees')
        total_angle += DEG_INT * angle * np.pi / 180
        positions.append(positions[-1] + arm['length'] * np.exp(1j * total_angle))
        
    # Check for intersections
    for i in range(len(positions) - 1):
        for j in range(i + 2, len(positions) - 1):  # i+2 to skip checking neighboring segments
            if segment_intersection(positions[i], positions[i+1], positions[j], positions[j+1]):
                return 0  # invalid configuration due to self-intersection

    return 1  # valid configuration
    
    
    
     


def calculate_valid_space(arm_config):

    num_arms = len(arm_config)
    
    # create c-space grid
    c_space = np.ones((DEG_STEP,) * num_arms, dtype=int)

    # invalidate all min-angle space
    for arm_idx, arm in enumerate(arm_config):
        min_angle = round((arm['angle-limit'] / 360) * DEG_STEP)

        slices = [slice(None)] * num_arms
        slices[arm_idx] = slice(0, min_angle)
        
        c_space[tuple(slices)] = 0
        
        
    # Get indices where c_space is 1
    indices = np.argwhere(c_space == 1)


    # Now, loop through these indices and validate
    for idx in indices:
        c_space[tuple(idx)] = validate(idx, arm_config)
    
        
    return c_space
    
    
    
import pyvista as pv
import matplotlib.colors as mcolors


def visualize_c_space_pyvista(c_space, start=None, end=None, path=None):
    if c_space.ndim != 3:
        print("Only 3D c_space can be visualized with this function.")
        return

    # Set up pyvista plotter
    plotter = pv.Plotter()

    # Create a grid object from the c_space
    grid = pv.UniformGrid()
    grid.dimensions = np.array(c_space.shape) + 1
    grid["values"] = c_space.flatten(order="F")

    # Show the valid (1s) regions
    threshed = grid.threshold([0.5, 1.5])
    plotter.add_mesh(threshed, show_edges=True, opacity=0.5, color="blue")

    # Highlight the start and end nodes
    if start and end:
        plotter.add_mesh(pv.Sphere(radius=0.5, center=start), color='green')
        plotter.add_mesh(pv.Sphere(radius=0.5, center=end), color='red')

    # Highlight the path
    if path:
        for point in path:
            plotter.add_mesh(pv.Sphere(radius=0.3, center=point), color='yellow')

    plotter.show()


    

    
if __name__ == "__main__":
    

    arm_config = [{'name': 'arm01', 'length': 4, 'angle-limit':10},
                 {'name': 'arm02', 'length': 1, 'angle-limit':10},
                 {'name': 'arm03', 'length': 4, 'angle-limit':10}]
    
    # calculate valid areas
    c_space = calculate_valid_space(arm_config)
    
    
    
    random_config_A = select_random_config(c_space, 1)
    
    random_config_B = select_random_config(c_space, 1)
    
    solver = AStarSolver(c_space)
    
    solution = solver.solve(random_config_A, random_config_B)
    
    
    print(solution)

    if solution:
        animate_solution(solution, arm_config, c_space)
    else:
        print("No solution found!")

    
    
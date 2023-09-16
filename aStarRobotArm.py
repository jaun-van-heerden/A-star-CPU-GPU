import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from aStar3d import AStarSolver
from aStarXd import AStarSolver

DEG_INT = 20
DEG_STEP = 360 // DEG_INT


class ArmAnimator:
    
    
    obstacle_segments = [
    (complex(1, 1), complex(3, 1)), 
    (complex(1, 2), complex(2, 1))
    ]
    
    
    
    def __init__(self, arm_config):
        self.arm_config = arm_config

    def animate_solution(self, solution):
        fig, ax = plt.subplots()

        total_len = sum([arm['length'] for arm in self.arm_config])
        ax.set_xlim(-total_len, total_len)
        ax.set_ylim(-total_len, total_len)
        ax.grid(True)
        ax.set_aspect('equal', 'box')

        line, = ax.plot([], [], 'o-', color='blue', label='Current Position')
        
        # Getting start and end endpoints for the arm
        start_x, start_y = self._config_to_endpoint(solution[0])
        end_x, end_y = self._config_to_endpoint(solution[-1])

        # Adding start and end markers for arm's endpoint
        start_marker, = ax.plot(start_x, start_y, 'go', markersize=8, label='Start Endpoint')
        end_marker, = ax.plot(end_x, end_y, 'ro', markersize=8, label='End Endpoint')
        
        
        # Draw obstacle segments
        for obs in obstacle_segments:
            plt.plot([obs[0].real, obs[1].real], [obs[0].imag, obs[1].imag], 'r-', linewidth=2)


        def init():
            line.set_data([], [])
            return line, start_marker, end_marker

        def update(frame):
            config = solution[frame]
            xdata, ydata = self._config_to_xy(config)
            line.set_data(xdata, ydata)
            
            # Ensure start and end markers are plotted at every frame
            start_marker.set_data(start_x, start_y)
            end_marker.set_data(end_x, end_y)

            return line, start_marker, end_marker 

        ani = FuncAnimation(fig, update, frames=len(solution), init_func=init, blit=True, repeat=False)
        ax.legend()
        plt.show()
        
    # def animate_solutions(self, solutions):
    #     fig, ax = plt.subplots()

    #     total_len = sum([arm['length'] for arm in self.arm_config])
    #     ax.set_xlim(-total_len, total_len)
    #     ax.set_ylim(-total_len, total_len)
    #     ax.grid(True)
    #     ax.set_aspect('equal', 'box')

    #     line, = ax.plot([], [], 'o-', color='blue', label='Current Position')

    #     # Flatten solutions into a single list for animation
    #     flat_solution = [config for solution in solutions for config in solution]

    #     def init():
    #         line.set_data([], [])
    #         return line,

    #     def update(frame):
    #         config = flat_solution[frame]
    #         xdata, ydata = self._config_to_xy(config)
    #         line.set_data(xdata, ydata)
    #         return line,

    #     ani = FuncAnimation(fig, update, frames=len(flat_solution), init_func=init, blit=True, repeat=False)
    #     ax.legend()
    #     plt.show()
        
        
        
    def animate_solutions(self, solutions):
        fig, ax = plt.subplots()

        total_len = sum([arm['length'] for arm in self.arm_config])
        ax.set_xlim(-total_len, total_len)
        ax.set_ylim(-total_len, total_len)
        ax.grid(True)
        ax.set_aspect('equal', 'box')

        line, = ax.plot([], [], 'o-', color='blue', label='Current Position')

        # Flatten solutions into a single list for animation
        flat_solution = [config for solution in solutions for config in solution]

        # Getting start and end endpoints for the arm
        start_x, start_y = self._config_to_endpoint(flat_solution[0])
        end_x, end_y = self._config_to_endpoint(flat_solution[-1])

        # Adding start and end markers for arm's endpoint
        start_marker, = ax.plot(start_x, start_y, 'go', markersize=8, label='Start Endpoint')
        end_marker, = ax.plot(end_x, end_y, 'ro', markersize=8, label='End Endpoint')
        
        # Draw obstacle segments
        for obs in self.obstacle_segments:
            plt.plot([obs[0].real, obs[1].real], [obs[0].imag, obs[1].imag], 'r-', linewidth=2)

        def init():
            line.set_data([], [])
            start_marker.set_data([], [])
            end_marker.set_data([], [])
            return line, start_marker, end_marker

        def update(frame):
            config = flat_solution[frame]
            xdata, ydata = self._config_to_xy(config)
            line.set_data(xdata, ydata)
            
            # Ensure start and end markers are plotted at every frame
            start_marker.set_data(start_x, start_y)
            end_marker.set_data(end_x, end_y)
            
            return line, start_marker, end_marker 

        ani = FuncAnimation(fig, update, frames=len(flat_solution), init_func=init, blit=True, repeat=False)
        ax.legend()
        plt.show()
        


    def _config_to_xy(self, config):
        total_angle = 0
        xdata, ydata = [0], [0]
        for angle, arm in zip(config, self.arm_config):
            total_angle += DEG_INT * angle * np.pi / 180
            end_x, end_y = xdata[-1] + arm['length'] * np.cos(total_angle), ydata[-1] + arm['length'] * np.sin(total_angle)
            xdata.append(end_x)
            ydata.append(end_y)
        return xdata, ydata

    def _config_to_endpoint(self, config):
        xdata, ydata = self._config_to_xy(config)
        return xdata[-1], ydata[-1]
    
    
    
    
class ArmConfiguration:
    

    
    @staticmethod
    def calculate_valid_space(arm_config):
        # This method should calculate and return the valid configuration space (c_space) for the given arm configuration.
        # This could involve iterating through all possible configurations, validating them, and then marking them as valid/invalid.
        
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
            if ArmConfiguration.self_intersect(idx, arm_config):
                c_space[tuple(idx)] = 0
        
        return c_space
    
    
        
    
    
    
    @staticmethod
    def plot_arm_configuration(config, arm_config):
        fig, ax = plt.subplots()
        total_angle = 0
        start_point = complex(0, 0)
        for angle, arm in zip(config, arm_config):
            total_angle += DEG_INT * angle * np.pi / 180
            end_point = start_point + arm['length'] * np.exp(1j * total_angle)
            ax.plot([start_point.real, end_point.real], [start_point.imag, end_point.imag], 'o-')
            start_point = end_point
        max_arm_length = sum([arm['length'] for arm in arm_config])
        ax.set_xlim(-max_arm_length, max_arm_length)
        ax.set_ylim(-max_arm_length, max_arm_length)
        ax.grid(True)
        ax.set_aspect('equal', 'box')
        plt.show()
        
        
        
        
        
        
    @staticmethod
    def intersects_obstacle(segment, obstacles):
        
        
        def ccw(A, B, C):
            return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)
        
        # Check if line segments AB and CD intersect
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        
        for obstacle in obstacles:
            if intersect(segment[0], segment[1], obstacle[0], obstacle[1]):
                return True
        return False
        
        
    @staticmethod
    def self_intersect(config, arm_config):
        
        def ccw(A, B, C):
            return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)

        # Check if line segments AB and CD intersect
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        

        # Generate the arm segments using the config and arm_config
        total_angle = 0
        start_point = complex(0, 0)
        segments = []
        for angle, arm in zip(config, arm_config):
            total_angle += DEG_INT * angle * np.pi / 180
            end_point = start_point + arm['length'] * np.exp(1j * total_angle)
            segments.append((start_point, end_point))
            start_point = end_point

        # Check if any two segments intersect
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)): # Changed offset here
                if i != j: # Ensuring we don't check a segment against itself
                    if intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                        return True
                    
            # Check if any segment intersects with obstacles
        for seg in segments:
            if ArmConfiguration.intersects_obstacle(seg, obstacle_segments):
                return True
        
        return False


    @staticmethod
    def validate(config, arm_config):
        if ArmConfiguration.self_intersect(config, arm_config):
            raise ValueError("Arm configuration has self-intersections")


def select_random_configs(c_space, val, count=1):
    indices = np.argwhere(c_space == val)
    if not indices.size:
        return None
    return [tuple(indices[np.random.choice(len(indices))]) for _ in range(count)]




if __name__ == "__main__":
    arm_config = [
        {'name': 'arm01', 'length': 1, 'angle-limit': 5},
        {'name': 'arm02', 'length': 1, 'angle-limit': 5},
        {'name': 'arm03', 'length': 1, 'angle-limit': 5},
        {'name': 'arm04', 'length': 1, 'angle-limit': 5}
    ]
    
    
    obstacle_segments = [
        (complex(1, 1), complex(3, 1)), 
        (complex(1, 2), complex(2, 1))
    ]

    

    c_space = ArmConfiguration.calculate_valid_space(arm_config)


    # random_config_A = select_random_config(c_space, 1)
    # random_config_B = select_random_config(c_space, 1)
    
    # answer = ArmConfiguration.self_intersect(random_config_B, arm_config)
    # print(answer)
    

    solver = AStarSolver(c_space)
    
    # solution = solver.solve(random_config_A, random_config_B)

    # if solution:
    #     print(solution)
    #     animator = ArmAnimator(arm_config)
    #     animator.animate_solution(solution)
    # else:
    #     print("No solution found!")
        
        
    num_random_configs = 5
    random_configs = select_random_configs(c_space, 1, num_random_configs)

    animator = ArmAnimator(arm_config)
    
    all_solutions = []
    current_config = random_configs[0]
    for next_config in random_configs[1:]:
        solution = solver.solve(current_config, next_config)
        if solution:
            all_solutions.append(solution)
            current_config = next_config  # set the end of this solution as the start for the next
        else:
            print(f"No solution found between {current_config} and {next_config}!")

    animator.animate_solutions(all_solutions)
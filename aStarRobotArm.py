import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from aStarXd import AStarSolver

STEP_INT = 5
DEG_STEP = 360 // STEP_INT



def plot_segments(segments):
    """
    Visualize a list of segments represented as complex numbers on a 2D plane.

    Parameters:
    - segments (list): A list of tuples. Each tuple contains two complex numbers representing the start and end of a segment.
    """
    # Plot each segment
    for seg in segments:
        plt.plot([seg[0].real, seg[1].real], [seg[0].imag, seg[1].imag], 'o-')

    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Visualization of Complex Number Segments')
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.show()







class ArmAnimator:

    def __init__(self, arm_config, obstacle_segments):
        self.arm_config = arm_config
        self.obstacle_segments = obstacle_segments
        
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
        start_point = ArmConfiguration.calculate_segments(flat_solution[0], arm_config)[-1][-1]
        end_point = ArmConfiguration.calculate_segments(flat_solution[-1], arm_config)[-1][-1]
        
        # start_x, start_y = self._config_to_endpoint(flat_solution[0])
        #target_points = [ArmConfiguration.calculate_segments(solution[-1], arm_config)[-1][-1] for solution in solutions]
        target_points = []
        for solution in solutions:
            target_points.extend([ArmConfiguration.calculate_segments(solution[-1], arm_config)[-1][-1]] * len(solution))
        
        

        # Adding start and end markers for arm's endpoint
        start_marker, = ax.plot(start_point.real, start_point.imag, 'go', markersize=4, label='Start Endpoint')
        end_marker, = ax.plot(end_point.real, end_point.imag, 'ro', markersize=4, label='End Endpoint')
        
        target_marker, = ax.plot(target_points[0].real, target_points[0].imag, 'yx', markersize=4, label='Target')
        
        # Draw obstacle segments
        for obs in self.obstacle_segments:
            plt.plot([obs[0].real, obs[1].real], [obs[0].imag, obs[1].imag], 'r-', linewidth=2)

        def init():
            line.set_data([], [])
            start_marker.set_data(start_point.real, start_point.imag)
            end_marker.set_data(end_point.real, end_point.imag)
            target_marker.set_data(target_points[0].real, target_points[0].imag)
            return line, start_marker, end_marker, target_marker

        def update(frame):
            config = flat_solution[frame]

            real_parts = []
            imag_parts = []
            for segment in ArmConfiguration.calculate_segments(config, arm_config):
                real_parts.append([c.real for c in segment])
                imag_parts.append([c.imag for c in segment])
            
            line.set_data(real_parts, imag_parts)
            target_point = target_points[frame]
            target_marker.set_data([target_point.real], [target_point.imag])
            
            return line, start_marker, end_marker, target_marker 

        ani = FuncAnimation(fig, update, frames=len(flat_solution), init_func=init, blit=True, repeat=False)
        ax.legend()
        plt.show()
        
        
    
class ArmConfiguration:
        
    @staticmethod
    def calculate_segments(config, arm_config):
        
        start_point = complex(0, 0)
        segments = []
        
        parent_angle = 0
        for angle, arm in zip(config, arm_config):
            angle_degrees = STEP_INT * angle
            angle_radians = parent_angle + (angle_degrees * np.pi / 180)
            end_point = start_point + arm['length'] * np.exp(1j * angle_radians)
            parent_angle = angle_radians - np.pi
            segments.append((start_point, end_point))
            start_point = end_point
        
        return segments
    
    
    
    
    @staticmethod
    def calculate_valid_space(arm_config):
        # This method should calculate and return the valid configuration space (c_space) for the given arm configuration.
        # This could involve iterating through all possible configurations, validating them, and then marking them as valid/invalid.
        
        num_arms = len(arm_config)
    
        # create c-space grid
        c_space = np.ones((DEG_STEP,) * num_arms, dtype=int)
            
        for arm_idx, arm in enumerate(arm_config):
            
            angle_limit = arm['angle-limit']
            
            min_angle = round((angle_limit / 360) * DEG_STEP)
            
            max_angle = DEG_STEP - min_angle

            slices = [slice(None)] * num_arms

            # Slice from the start of the array up to min_angle (not inclusive)
            slices[arm_idx] = slice(0, min_angle)
            c_space[tuple(slices)] = 0
            
            # Slice from max_angle to the end of the array
            slices[arm_idx] = slice(max_angle, None)
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
        
        for segment in ArmConfiguration.calculate_segments(config, arm_config):
            real_parts = [c.real for c in segment]
            imag_parts = [c.imag for c in segment]
            ax.plot(real_parts, imag_parts, 'o-')
        
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
        
        segments = ArmConfiguration.calculate_segments(config, arm_config)
                    
        # Check if any two segments intersect
        for i in range(len(segments) - 1):  # no need to check the last segment against others
            for j in range(i + 2, len(segments)):  # Start from i+2 to skip the next consecutive segment
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




def visualize_c_space_slice(c_space, joint1, joint2):
    """
    Visualizes the configuration space for two specified joints.
    """
    # Extracting the 2D slice for the given joints
    c_space_slice = c_space.sum(axis=tuple([i for i in range(c_space.ndim) if i not in [joint1, joint2]]))
    
    # Plotting
    plt.imshow(c_space_slice, cmap='gray_r', interpolation='none', origin='lower')
    plt.colorbar(label='Number of valid configurations')
    plt.title(f"C-Space for Joint {joint1 + 1} and Joint {joint2 + 1}")
    plt.xlabel(f"Joint {joint1 + 1} angle (increments of {STEP_INT} degrees)")
    plt.ylabel(f"Joint {joint2 + 1} angle (increments of {STEP_INT} degrees)")
    plt.show()




if __name__ == "__main__":
    arm_config = [
        {'name': 'arm01', 'length': 1, 'angle-limit': 10},
        {'name': 'arm02', 'length': 1, 'angle-limit': 10},
        {'name': 'arm03', 'length': 1, 'angle-limit': 10}
        # {'name': 'arm04', 'length': 1, 'angle-limit': 5}
    ]
    
    
    obstacle_segments = [
        (complex(-2, 1), complex(-2, 0)), 
        (complex(-1, -2), complex(-1, -2)),
        (complex(1, 1), complex(3, 1))
        
    ]
    # obstacle_segments = [

    # ]
    
    # safety_margin = 0.2  # for example
    # expanded_obstacles = [seg for obstacle in obstacle_segments for seg in expand_obstacle(obstacle, safety_margin)]
    # obstacle_segments = obstacle_segments + expanded_obstacles



    

    c_space = ArmConfiguration.calculate_valid_space(arm_config)
    
    
    # Usage example: visualize the c_space slice for joints 0 and 1
    visualize_c_space_slice(c_space, 0, 1)

    solver = AStarSolver(c_space)
        
        
    num_random_configs = 10
    random_configs = select_random_configs(c_space, 1, num_random_configs)
    
    print(random_configs)
    
    # random_configs = [(30//STEP_INT, 60//STEP_INT, 60//STEP_INT), 
    #                   (20//STEP_INT, 340//STEP_INT, 20//STEP_INT),
    #                   (340//STEP_INT, 20//STEP_INT, 20//STEP_INT),
    #                   (20//STEP_INT, 20//STEP_INT, 340//STEP_INT),
    #                   (340//STEP_INT, 340//STEP_INT, 20//STEP_INT),
    #                   (20//STEP_INT, 340//STEP_INT, 340//STEP_INT),
    #                   (340//STEP_INT, 20//STEP_INT, 340//STEP_INT),
    #                   (340//STEP_INT, 340//STEP_INT, 340//STEP_INT)]
    

    #ArmConfiguration.self_intersect(random_configs[0], arm_config)

    ArmConfiguration.plot_arm_configuration(random_configs[0], arm_config)
    
    
    

    animator = ArmAnimator(arm_config, obstacle_segments)
    
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
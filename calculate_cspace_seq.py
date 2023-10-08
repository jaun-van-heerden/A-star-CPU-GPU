import numpy as np
import pstats
from io import StringIO

def ccw(A, B, C):
    """
    Determines whether three points are in counterclockwise order.
    
    Args:
        A (complex): First point.
        B (complex): Second point.
        C (complex): Third point.
        
    Returns:
        bool: True if in counterclockwise order, False otherwise.
    """
    return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)


def intersect(A, B, C, D):
    """
    Checks if two line segments AB and CD intersect.
    
    Args:
        A (complex): Start point of segment AB.
        B (complex): End point of segment AB.
        C (complex): Start point of segment CD.
        D (complex): End point of segment CD.
        
    Returns:
        bool: True if the segments intersect, False otherwise.
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def calculate_segments(config, setup):
    """
    Calculate line segments representing a robotic arm's configuration.
    
    Args:
        config (list): List of joint angles.
        setup (dict): Configuration settings.
        
    Returns:
        list: List of line segments as (start, end) complex numbers.
    """

    start_point = complex(0, 0)
    segments = []
    
    total_angle = 0
    for angle in config:
        angle_degrees = setup["step_int"] * angle
    
        total_angle = total_angle + (angle_degrees * np.pi / 180)
        
        end_point = start_point + np.exp(1j * total_angle)
    
        segments.append((start_point, end_point))
        start_point = end_point
    
    return segments


def self_intersect(config, setup):
    """
    Check if a robotic arm's configuration intersects with itself.
    
    Args:
        config (list): List of joint angles.
        setup (dict): Configuration settings.
        
    Returns:
        bool: True if self-intersecting, False otherwise.
    """   
    segments = calculate_segments(config, setup)
                
    for i in range(len(segments) - 1):
        for j in range(i + 2, len(segments)):  # Start from i+2 to skip checking consecutive arms
            if intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                return True

    return False


def calculate_cspace(setup):
    """
    Calculate the configuration space (C-Space) for a robotic arm.
    
    Args:
        setup (dict): Configuration settings.
        
    Returns:
        numpy.ndarray: C-Space grid.
    """
    num_arms = len(setup["arm_config"])

    # create c-space grid
    c_space = np.ones((setup["deg_step"],) * num_arms, dtype=int)

    # Get indices where c_space is 1
    indices = np.argwhere(c_space == 1)
    
    for idx in indices:
        if self_intersect(idx, setup):
            c_space[tuple(idx)] = 0

    return c_space


if __name__ == "__main__":
    
    import cProfile

    for STEP in [64, 32, 16, 8, 4, 2, 1]:

        test_setup = {
            "arm_config" :[
                {'angle-limit': 10},
                {'angle-limit': 10},
                {'angle-limit': 10}
            ],
            "step_int": STEP,
            "deg_step": 360//STEP
        }
        
        pr = cProfile.Profile()
        pr.enable()

        cspace = calculate_cspace(test_setup)

        pr.disable()
        pr.dump_stats(f"profiles/profile_seq_{STEP}.prof")

        np.save(f'cspaces/cspace_seq_{STEP}.npy', cspace)

        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    
    
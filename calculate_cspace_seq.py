import numpy as np
from hash_func import hash_cspace
import pstats
from io import StringIO

def ccw(A, B, C):
    return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)

# Check if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def calculate_segments(config, setup):

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
    
    segments = calculate_segments(config, setup)
                
    for i in range(len(segments) - 1):
        for j in range(i + 2, len(segments)):  # Start from i+2 to skip checking consecutive arms
            if intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                return True

    return False


def validate(config, setup):
    if self_intersect(config, setup):
        raise ValueError("Arm configuration has self-intersections")


def calculate_cspace(setup):
    
    num_arms = len(setup["arm_config"])

    # create c-space grid
    c_space = np.ones((setup["deg_step"],) * num_arms, dtype=int)

    for arm_idx, arm in enumerate(setup["arm_config"]):
        angle_limit = arm['angle-limit']
        min_angle = round((angle_limit / 360) * setup["deg_step"])
        max_angle = setup["deg_step"] - min_angle
        c_space[arm_idx, 0:min_angle] = 0
        c_space[arm_idx, max_angle:] = 0

        
    # Get indices where c_space is 1
    indices = np.argwhere(c_space == 1)
    
    for idx in indices:
        if self_intersect(idx, setup):
            c_space[tuple(idx)] = 0

    return c_space


if __name__ == "__main__":
    
    import cProfile

    STEP = 10

    test_setup = {
        "arm_config" :[
            {'angle-limit': 10},
            {'angle-limit': 10},
            {'angle-limit': 10}
        ],
        "step_int": STEP,
        "deg_step": 360//STEP
    }
    
    
    from display_segment import display_robot_arm
    
    # test
    config = [35,27,15]
    
    result = calculate_segments(config, test_setup)
    
    print(result)
    display_robot_arm(result)
    
  

    pr = cProfile.Profile()
    pr.enable()

    cspace = calculate_cspace(test_setup)

    pr.disable()
    pr.dump_stats(f"profile_seq_{STEP}.prof")

    print(hash_cspace(cspace))

    np.save('cspace_seq.npy', cspace)

    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    #print(s.getvalue())
    


    import random
    # Step 1: Randomly select a C-Space point with zero value
    zero_indices = np.argwhere(cspace == 0)
    if len(zero_indices) == 0:
        print("No zero-valued points in the C-Space.")
    else:
        random_zero_idx = random.choice(zero_indices)
        print(f"Randomly selected zero-valued C-Space point: {random_zero_idx}")
        
        # Step 2: Plot using display_robot_arm
        random_segments = calculate_segments(random_zero_idx, test_setup)
        print(f"Segments for the randomly selected zero-valued C-Space point: {random_segments}")
        display_robot_arm(random_segments)
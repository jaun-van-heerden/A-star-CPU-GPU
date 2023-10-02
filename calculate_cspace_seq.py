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
    
    parent_angle = 0
    for angle, arm in zip(config, setup["arm_config"]):
        angle_degrees = setup["step_int"] * angle
        angle_radians = parent_angle + (angle_degrees * np.pi / 180)
        end_point = start_point + arm['length'] * np.exp(1j * angle_radians)
        parent_angle = angle_radians - np.pi
        segments.append((start_point, end_point))
        start_point = end_point
    
    return segments


def self_intersect(config, setup):
    
    segments = calculate_segments(config, setup)
                
    # Check if any two segments intersect
    for i in range(len(segments) - 1):  # no need to check the last segment against others
        for j in range(i + 2, len(segments)):  # Start from i+2 to skip the next consecutive segment
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

    # for arm_idx, arm in enumerate(setup["arm_config"]):
    #     angle_limit = arm['angle-limit']
    #     min_angle = round((angle_limit / 360) * setup["deg_step"])
    #     max_angle = setup["deg_step"] - min_angle
    #     c_space[arm_idx, 0:min_angle] = 0
    #     c_space[arm_idx, max_angle:] = 0

        
    # Get indices where c_space is 1
    indices = np.argwhere(c_space == 1)
    
    for idx in indices:
        if self_intersect(idx, setup):
            c_space[tuple(idx)] = 0

    return c_space


if __name__ == "__main__":
    
    import cProfile

    STEP = 90

    test_setup = {
        "arm_config" :[
            {'name': 'arm01', 'length': 1, 'angle-limit': 10},
            {'name': 'arm02', 'length': 1, 'angle-limit': 10},
            {'name': 'arm03', 'length': 1, 'angle-limit': 10}
        ],
        "step_int": STEP,
        "deg_step": 360//STEP,
        "degrees_to_radians": 0.01745329251 
    }

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

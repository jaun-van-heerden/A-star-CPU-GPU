import numpy as np
#from functools import partial
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from hash_func import hash_cspace



def ccw(A, B, C):
    return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)


# Check if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)





# MAIN MULTIPROCESS FUNCTION


def validate_config(args):
    indices_chunk, setup = args  # Unpack the tuple
    results_chunk = []
    for idx in indices_chunk:
        value = int(not self_intersect(idx, setup))  # Your current logic here
        results_chunk.append((idx, value))
    return results_chunk

# def validate_config(indices_chunk, setup):
#     results_chunk = []
#     for idx in indices_chunk:
#         value = int(not self_intersect(idx, setup)) # Your current logic here
#         results_chunk.append((idx, value))
#     return results_chunk


def calculate_segments(config, setup):
    
    start_point = complex(0, 0)
    segments = []
    
    parent_angle = 0
    for angle, arm in zip(config, setup["arm_config"]):
        angle_degrees = setup["step_int"] * angle
        angle_radians = parent_angle + (angle_degrees * setup["degrees_to_radians"])
        end_point = start_point + arm['length'] * np.exp(1j * angle_radians)
        parent_angle = angle_radians - np.pi
        segments.append((start_point, end_point))
        start_point = end_point
    
    return segments


def intersects_obstacle(segment, setup):
    
    for obstacle in setup["obstacle_config"]:
        if intersect(segment[0], segment[1], obstacle[0], obstacle[1]):
            return True

    return False
    

def self_intersect(config, setup):
    
    segments = calculate_segments(config, setup)
                
    # Check if any two segments intersect
    for i in range(len(segments) - 1):  # no need to check the last segment against others
        for j in range(i + 2, len(segments)):  # Start from i+2 to skip the next consecutive segment
            if intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                return True

    # Check if any segment intersects with obstacles
    for seg in segments[1:]:   # we dont need to check the first one
        if intersects_obstacle(seg, setup):
            return True
    
    return False


def calculate_cspace_parallel(setup):
    
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
    
    indices_chunks = np.array_split(indices, cpu_count())

    # with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    #     results_chunks = list(executor.map(partial(wrapped_validate_config, setup), indices_chunks))

    # Here, we're zipping setup with each indices_chunk to create a tuple
    args_list = [(chunk, setup) for chunk in indices_chunks]

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results_chunks = list(executor.map(validate_config, args_list))

    # Flatten the results and populate the c_space
    for results in results_chunks:
        for idx, value in results:
            c_space[tuple(idx)] = value

    return c_space


if __name__ == "__main__":

    import cProfile

    STEP = 10
    
    test_setup = {
        "arm_config" :[
            {'name': 'arm01', 'length': 1, 'angle-limit': 10},
            {'name': 'arm02', 'length': 1, 'angle-limit': 10},
            {'name': 'arm03', 'length': 1, 'angle-limit': 10}
        ],
        "obstacle_config": [
            (complex(-2, 1), complex(-2, 0)), 
            (complex(-1, -2), complex(-1, -2)),
            (complex(1, 1), complex(3, 1))
        ],
        "step_int": STEP,
        "deg_step": 360//STEP,
        "degrees_to_radians": 0.01745329251 
    }

    pr = cProfile.Profile()
    pr.enable()
    
    calculate_cspace_parallel(test_setup)

    pr.disable()
    pr.dump_stats(f"profile_par_{STEP}.prof")
        
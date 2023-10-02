import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from hash_func import hash_cspace
import pstats
from io import StringIO


def ccw_vec(A, B, C):
    return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)

def intersect_vec(A, B, C, D):
    return np.logical_and(ccw_vec(A, C, D) != ccw_vec(B, C, D),
                          ccw_vec(A, B, C) != ccw_vec(A, B, D))

# def calculate_segments_vec(config, setup):
#     num_arms = config.shape[1]
#     segments = np.zeros((config.shape[0], num_arms, 2), dtype=np.complex_)
#     start_point = np.zeros((config.shape[0],), dtype=np.complex_)
#     parent_angle = np.zeros((config.shape[0],))
    
#     for i in range(num_arms):
#         arm = setup["arm_config"][i]
#         angle_degrees = setup["step_int"] * config[:, i]
#         angle_radians = parent_angle + angle_degrees * setup["degrees_to_radians"]
#         end_point = start_point + arm['length'] * np.exp(1j * angle_radians)
#         segments[:, i, :] = np.column_stack((start_point, end_point))
#         parent_angle = angle_radians - np.pi
#         start_point = end_point
#     return segments


def calculate_segments_vec(config, setup):
    num_arms = config.shape[1]
    num_configs = config.shape[0]
    segments = np.zeros((num_configs, num_arms, 2), dtype=np.complex_)
    start_point = np.zeros((num_configs,), dtype=np.complex_)
    parent_angle = np.zeros((num_configs,))

    lengths = np.array([arm['length'] for arm in setup["arm_config"]])
    step_int = setup["step_int"]
    degrees_to_radians = setup["degrees_to_radians"]

    angle_degrees = step_int * config
    angle_radians = np.cumsum(angle_degrees, axis=1) * degrees_to_radians

    end_points = lengths * np.exp(1j * angle_radians)

    segments[:, :, 1] = np.cumsum(end_points, axis=1)
    segments[:, 1:, 0] = segments[:, :-1, 1]

    return segments

# No changes needed in other parts of the code







def validate_config_vec(args):
    indices_chunk, setup = args
    results_chunk = []
    configs = indices_chunk  # Assuming indices directly relate to config
    segments = calculate_segments_vec(configs, setup)
    
    for idx, seg_set in zip(indices_chunk, segments):
        intersect_results = np.array([
            intersect_vec(A, B, C, D)
            for A, B in seg_set[:-1]
            for C, D in seg_set[1:]
        ])
        self_intersect = np.any(intersect_results)
        results_chunk.append((tuple(idx), int(not self_intersect)))
    return results_chunk




def calculate_cspace_parallel(setup):
    
    num_arms = len(setup["arm_config"])

    # create c-space grid
    c_space = np.ones((setup["deg_step"],) * num_arms, dtype=int)



        
    # Get indices where c_space is 1
    #indices = np.argwhere(c_space == 1)
    
    indices_chunks = np.array_split(c_space, cpu_count())

    # Here, we're zipping setup with each indices_chunk to create a tuple
    args_list = [(chunk, setup) for chunk in indices_chunks]

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results_chunks = list(executor.map(validate_config_vec, args_list))

    # Flatten the results and populate the c_space
    for results in results_chunks:
        for idx, value in results:
            c_space[tuple(idx)] = value


    for arm_idx, arm in enumerate(setup["arm_config"]):
        angle_limit = arm['angle-limit']
        min_angle = round((angle_limit / 360) * setup["deg_step"])
        max_angle = setup["deg_step"] - min_angle
        c_space[arm_idx, 0:min_angle] = 0
        c_space[arm_idx, max_angle:] = 0

    return c_space



if __name__ == "__main__":

    import cProfile

    STEP = 5
    
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
    
    cspace = calculate_cspace_parallel(test_setup)

    pr.disable()
    pr.dump_stats(f"profile_par_{STEP}.prof")

    print(hash_cspace(cspace))

    np.save('cspace_par.npy', cspace)

    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    #print(s.getvalue())
        
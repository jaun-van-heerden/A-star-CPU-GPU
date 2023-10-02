import numpy as np
from hash_func import hash_cspace
import pstats
from io import StringIO


# def generate_indices(shape):
#     grid = np.meshgrid(*[np.arange(dim) for dim in shape], indexing='ij')
#     return np.stack(grid, axis=-1)

# def ccw_vec(A, B, C):
#     return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)

# def intersect_vec(A, B, C, D):
#     return np.logical_and(ccw_vec(A, C, D) != ccw_vec(B, C, D),
#                           ccw_vec(A, B, C) != ccw_vec(A, B, D))

# def calculate_segments_vec(configs, setup):
#     num_arms = len(setup["arm_config"])
#     segments = np.zeros((num_arms, 2), dtype=np.complex_)
#     start_point = 0j
#     parent_angle = 0

#     for i, (angle_radians, arm) in enumerate(zip(configs, setup["arm_config"])):
#         #angle_degrees = setup["step_int"] * angle
#         angle_radians = parent_angle + angle_radians #np.radians(angle_degrees)
#         end_point = start_point + arm['length'] * np.exp(1j * angle_radians)
#         segments[i, :] = (start_point, end_point)
#         parent_angle = angle_radians - np.pi
#         start_point = end_point

#     return segments

# def self_intersect_vec(config, setup):
#     segments = calculate_segments_vec(config, setup)
#     A, B = segments[:-1, 0], segments[:-1, 1]
#     C, D = segments[1:, 0], segments[1:, 1]
#     intersect_results = intersect_vec(A, B, C, D)
#     return np.any(intersect_results)


# def self_intersect_batch(configs, setup):
#     intersects = np.zeros(configs.shape[0], dtype=bool)
#     for i, config in enumerate(configs):
#         segments = calculate_segments_vec(config, setup)
#         A, B = segments[:-1, 0], segments[:-1, 1]
#         C, D = segments[1:, 0], segments[1:, 1]
#         intersects[i] = np.any(intersect_vec(A, B, C, D))
#     return intersects



# # Vectorized self_intersect function
# def vectorized_self_intersect(configs, setup):
#     # Perform your collision logic here
#     calculate_segments_vec(configs, setup)

def ccw(A, B, C):
    return (np.imag(C) - np.imag(A)) * (np.real(B) - np.real(A)) > (np.imag(B) - np.imag(A)) * (np.real(C) - np.real(A))

def check_intersection(segments):
    num_arms = segments.shape[-2]
    no_intersect = np.ones(segments.shape[:-2], dtype=np.int8)

    for i in range(num_arms):
        for j in range(i + 1, num_arms):
            A, B = segments[..., i, 0], segments[..., i, 1]
            C, D = segments[..., j, 0], segments[..., j, 1]
            
            # Use the ccw function in a vectorized manner
            ccw_cond1 = ccw(A, C, D) != ccw(B, C, D)
            ccw_cond2 = ccw(A, B, C) != ccw(A, B, D)

            # Find where both conditions are True
            intersects = np.logical_and(ccw_cond1, ccw_cond2)

            # Update the no_intersect array based on the intersects array
            no_intersect[intersects] = 0

    return no_intersect


def segments_vec(mesh):
    num_arms = mesh.shape[-1]

    # Create an array to hold segments with the same shape as the mesh, but with an added dimension for segment endpoints
    segments_shape = mesh.shape + (2,)
    segments = np.zeros(segments_shape, dtype=np.complex_)

    # Calculate the cumulative sum of angles along the last axis (i.e., for each arm)
    angle_radians_cumsum = np.cumsum(mesh, axis=-1, dtype=np.float64)

    # Adjust for parent_angle_offsets using broadcasting
    parent_angle_offsets = np.pi * np.arange(num_arms)
    angle_radians_cumsum -= parent_angle_offsets[np.newaxis, np.newaxis, np.newaxis, :]

    # Calculate the end points for each segment
    end_points = np.exp(1j * angle_radians_cumsum).cumsum(axis=-1)

    # Fill in the segment coordinates
    segments[..., 0] = np.concatenate([np.zeros(mesh.shape[:-1] + (1,), dtype=np.complex_), end_points[..., :-1]], axis=-1)
    segments[..., 1] = end_points

    return segments


# def segments_vec(config):
#     num_arms = len(config)
#     segments = np.zeros((num_arms, 2), dtype=np.complex_)

#     # Compute the cumulative sum of angles in a more memory-efficient manner
#     angle_radians_cumsum = np.cumsum(config, dtype=np.float64)

#     # Use in-place operations to adjust for parent_angle_offsets
#     parent_angle_offsets = np.pi * np.arange(num_arms)
#     np.subtract(angle_radians_cumsum, parent_angle_offsets, out=angle_radians_cumsum)

#     # Compute end_points with cumulative sum to save memory
#     end_points = np.exp(1j * angle_radians_cumsum).cumsum()

#     # Directly populate the segments array without extra array creation
#     segments[:, 0] = np.concatenate(([0j], end_points[:-1]))
#     segments[:, 1] = end_points

#     return segments





def calculate_cspace_vec(setup):

    num_arms = len(setup["arm_config"])

    c_space_dim = (setup["deg_step"],) * num_arms
    c_space = np.ones(c_space_dim, dtype=int)

    slices = setup["deg_step"]

    slice_objects = [slice(0, slices) for _ in range(num_arms)]

    grid = np.mgrid[slice_objects] * 0.0174533 # angle already in radians


    # # Generate meshgrid indices
    mesh = np.meshgrid(np.arange(c_space.shape[0]), 
                                        np.arange(c_space.shape[1]), 
                                        np.arange(c_space.shape[2]), 
                                        indexing='ij')

    index_tuples = np.stack(mesh, axis=-1) * 0.0174533

    segments = segments_vec(index_tuples)
    c_space = check_intersection(segments)


    # # Angle limit application
    # for arm_idx, arm in enumerate(setup["arm_config"]):
    #     angle_limit = arm['angle-limit']
    #     min_angle = round((angle_limit / 360) * setup["deg_step"])
    #     max_angle = setup["deg_step"] - min_angle
    #     c_space[arm_idx, 0:min_angle] = 0
    #     c_space[arm_idx, max_angle:] = 0

    # Get indices where c_space is 1
    #indices = np.argwhere(c_space == 1)

    return c_space

if __name__ == "__main__":

    import cProfile

    STEP = 90
    test_setup = {
        "arm_config": [
            {'name': 'arm01', 'length': 1, 'angle-limit': 10},
            {'name': 'arm02', 'length': 1, 'angle-limit': 10},
            {'name': 'arm03', 'length': 1, 'angle-limit': 10}
        ],
        "step_int": STEP,
        "deg_step": 360 // STEP
    }

    pr = cProfile.Profile()
    pr.enable()

    cspace = calculate_cspace_vec(test_setup)

    pr.disable()
    pr.dump_stats(f"profile_seq_{STEP}.prof")

    print(hash_cspace(cspace))

    np.save('cspace_vec.npy', cspace)

    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

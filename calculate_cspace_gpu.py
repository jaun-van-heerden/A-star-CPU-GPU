import cupy as np
import pstats
from io import StringIO


def ccw(A, B, C):
    """
    Counter-Clockwise Test
    
    Parameters:
        A, B, C: complex numbers representing points in 2D plane
    
    Returns:
        bool: True if points A, B, and C are in counter-clockwise order, False otherwise
    """
    return (np.imag(C) - np.imag(A)) * (np.real(B) - np.real(A)) > (np.imag(B) - np.imag(A)) * (np.real(C) - np.real(A))


def check_intersection(segments):
    num_arms = segments.shape[-2]
    no_intersect = np.ones(segments.shape[:-2], dtype=np.int8)

    for i in range(num_arms):
        for j in range(i + 2, num_arms):  # Start from i+2 to skip checking consecutive arms

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
    """
    Generate arm segments from a mesh grid.
    
    Parameters:
        mesh : cupy.ndarray
            Mesh grid array.
    
    Returns:
        cupy.ndarray : Array containing segments.
    """
    segments_shape = mesh.shape + (2,)
    segments = np.zeros(segments_shape, dtype=np.complex128)

    angle_degrees_cumsum = np.cumsum(mesh, axis=-1, dtype=np.float64) 
    angle_radians_cumsum = angle_degrees_cumsum * (np.pi / 180)

    end_points = np.exp(1j * angle_radians_cumsum).cumsum(axis=-1)

    segments[..., 0] = np.concatenate([np.zeros(mesh.shape[:-1] + (1,), dtype=np.complex128), end_points[..., :-1]], axis=-1)
    segments[..., 1] = end_points
    
    return segments



def calculate_cspace_vec(setup):
    """
    Calculate configuration space.
    
    Parameters:
        setup : dict
            Dictionary containing setup information.
            
    Returns:
        cupy.ndarray, cupy.ndarray : Configuration space and segments.
    """
    num_arms = len(setup["arm_config"])

    c_space_dim = (setup["deg_step"],) * num_arms
    c_space = np.ones(c_space_dim, dtype=int)

    # Generate meshgrid indices
    mesh = np.meshgrid(np.arange(c_space.shape[0]), 
                                        np.arange(c_space.shape[1]), 
                                        np.arange(c_space.shape[2]), 
                                        indexing='ij')

    index_tuples = np.stack(mesh, axis=-1) * setup["step_int"] 

    segments = segments_vec(index_tuples)
    c_space = check_intersection(segments)

    return c_space, segments


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
        cspace, segments = calculate_cspace_vec(test_setup)
        pr.disable()
        
        pr.dump_stats(f"profiles/profile_gpu_{STEP}.prof")

        np.save(f'cspaces/cspace_gpu_{STEP}.npy', cspace)

        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        
        print(s.getvalue())


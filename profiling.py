import cProfile
import pstats
from io import StringIO

from aStarRobotArm_no_amin import calculate_cspace
from aStarRobotArm_parallel_no_amin import calculate_cspace_parallel


# from aStarRobotArm import ArmConfiguration

# from aStarRobotArm_parallel import ArmConfiguration as ArmParallel


# The original code goes here (from the `import numpy as np` till the end of your provided code)

if __name__ == "__main__":
    
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
        "step_int": 1,
        "deg_step": 360//1,
        "degrees_to_radians": 0.01745329251 
    }

    
    pr = cProfile.Profile()
    pr.enable()
    
    calculate_cspace_parallel(test_setup)
    
    pr.disable()
    
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    
    
    
    pr = cProfile.Profile()
    pr.enable()
    
    calculate_cspace(test_setup)
    
    pr.disable()
    
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
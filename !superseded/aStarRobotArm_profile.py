import cProfile
import pstats
from io import StringIO

from aStarRobotArm import ArmConfiguration

from aStarRobotArm_parallel import ArmConfiguration as ArmParallel


# The original code goes here (from the `import numpy as np` till the end of your provided code)

def main(arm):
    # Sample test calls of your functions can be placed here for profiling.
    # For instance:
    
    c_space = arm.calculate_valid_space()
    
    return c_space
    
    # Add more calls as required to cover all the functionalities

if __name__ == "__main__":
    
    
    arm_config = [
    {'name': 'arm01', 'length': 1, 'angle-limit': 10},
    {'name': 'arm02', 'length': 1, 'angle-limit': 10},
    {'name': 'arm03', 'length': 1, 'angle-limit': 10}
    ]
    
    
    obstacle_config = [
        (complex(-2, 1), complex(-2, 0)), 
        (complex(-1, -2), complex(-1, -2)),
        (complex(1, 1), complex(3, 1))
    ]
    
    arm = ArmConfiguration(arm_config, obstacle_config)
    
    
    pr = cProfile.Profile()
    pr.enable()
    
    main(arm)  # Execute the main function
    
    pr.disable()
    
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    



    arm = ArmParallel(arm_config, obstacle_config)
    
    
    pr = cProfile.Profile()
    pr.enable()
    
    main(arm)  # Execute the main function
    
    pr.disable()
    
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

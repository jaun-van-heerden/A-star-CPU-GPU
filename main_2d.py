from robot_arm import ArmConfiguration, ArmAnimator2D, DEG_STEP

arm_config = [
    {'name': 'arm01', 'length': 1, 'angle-limit': 10},
    {'name': 'arm02', 'length': 1, 'angle-limit': 10},
    {'name': 'arm03', 'length': 1, 'angle-limit': 10},
]

obstacle_segments = [
    (complex(-2, 1), complex(-2, 0)),
    (complex(-1, -2), complex(-1, -2)),
    (complex(1, 1), complex(3, 1)),
]

arm = ArmConfiguration(arm_config, obstacle_segments)
solver = arm.make_solver(DEG_STEP)

waypoints = [
    (32, 62, 21), (33, 26, 51), (30, 18, 34), (60, 45, 58), (40, 51, 65),
    (5, 7, 22), (54, 57, 36), (40, 68, 34), (57, 58, 43), (22, 26, 38),
]

animator = ArmAnimator2D(arm)
solutions = []
current = waypoints[0]
for nxt in waypoints[1:]:
    path = solver.solve(current, nxt)
    if path:
        solutions.append(path)
        current = nxt
    else:
        print(f"No path: {current} → {nxt}")

animator.animate_solutions(solutions)

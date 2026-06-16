import numpy as np
from robot_arm import ArmConfiguration3D, ArmAnimator3D, DEG_STEP

# Each joint: azimuth + elevation angle, both bounded by angle-limit.
# Config tuples have length 2 * n_links: (θ0, φ0, θ1, φ1, ...).
arm_config = [
    {'name': 'arm01', 'length': 1, 'angle-limit': 10},
    {'name': 'arm02', 'length': 1, 'angle-limit': 10},
    {'name': 'arm03', 'length': 1, 'angle-limit': 10},
]

# 3D obstacle segments: pairs of (x, y, z) numpy arrays
obstacle_segments_3d = [
    (np.array([-2.0, 0.0, 0.5]), np.array([-2.0, 0.0, -0.5])),
    (np.array([1.0, 1.0, 0.0]), np.array([3.0, 1.0, 0.0])),
]

arm = ArmConfiguration3D(arm_config, obstacle_segments_3d)
solver = arm.make_solver(DEG_STEP)

# Waypoints: (θ0, φ0, θ1, φ1, θ2, φ2) — all indices in [min_angle, max_angle)
# With angle-limit=10 and DEG_STEP=180: min_angle=5, valid range [5, 175)
waypoints = [
    (32, 45, 62, 50, 21, 40),
    (33, 40, 26, 55, 51, 35),
    (30, 50, 18, 45, 34, 60),
    (60, 35, 45, 50, 58, 40),
    (40, 55, 51, 40, 65, 45),
]

animator = ArmAnimator3D(arm)
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

import random
from robot_arm import ArmConfiguration, ArmAnimator2D, DEG_STEP

N_LINKS = 5          # change freely; higher = richer motion but slower per-solve
LINK_LENGTH = 0.7    # total reach = N_LINKS * LINK_LENGTH
WAYPOINT_SPREAD = 20 # max Chebyshev steps between consecutive waypoints
N_WAYPOINTS = 12
# WA* weight: >1 finds sub-optimal paths but exponentially faster in high-D.
# weight=1 is exact-optimal (fine for ≤3 links); weight=3 is good for 4-6 links.
WEIGHT = 1.0 if N_LINKS <= 3 else 5.0

arm_config = [
    {'name': f'arm{i:02d}', 'length': LINK_LENGTH, 'angle-limit': 10}
    for i in range(1, N_LINKS + 1)
]

# Obstacle map: (complex, complex) = line segment; (complex, float) = circle pillar.
# Arm is based at origin with reach = N_LINKS * LINK_LENGTH = 3.5
obstacle_config = [
    # Vertical wall on the right with a 1.6-unit gap centred at the origin
    (complex(2.0,  3.5), complex(2.0,  0.8)),
    (complex(2.0, -0.8), complex(2.0, -3.5)),
    # Horizontal shelf on the upper-left
    (complex(-3.5, 1.8), complex(-0.8, 1.8)),
    # Circle pillars
    (complex( 1.2, 2.4), 0.4),   # upper-right
    (complex(-1.8, -1.5), 0.4),  # lower-left
]

arm = ArmConfiguration(arm_config, obstacle_config)
solver = arm.make_solver(DEG_STEP)

min_a = [round((a['angle-limit'] / 360) * DEG_STEP) for a in arm_config]
max_a = [DEG_STEP - m for m in min_a]


def next_waypoint(base):
    for _ in range(3000):
        cfg = tuple(
            max(lo, min(hi - 1, x + random.randint(-WAYPOINT_SPREAD, WAYPOINT_SPREAD)))
            for x, lo, hi in zip(base, min_a, max_a)
        )
        if not arm.self_intersect(cfg):
            return cfg
    return None


print(f"Building {N_LINKS}-link arm waypoint chain...")
waypoints = [arm.random_valid_config(DEG_STEP)]
fails = 0
while len(waypoints) < N_WAYPOINTS:
    nxt = next_waypoint(waypoints[-1])
    if nxt:
        waypoints.append(nxt)
        fails = 0
    else:
        fails += 1
        if fails >= 20:
            # stuck near obstacles — jump to a new random valid config
            nxt = arm.random_valid_config(DEG_STEP)
            if nxt:
                waypoints.append(nxt)
            fails = 0
print(f"  {len(waypoints)} waypoints ready.")

print("Solving paths...")
solutions = []
current = waypoints[0]
for i, nxt in enumerate(waypoints[1:], 1):
    path = solver.solve(current, nxt, weight=WEIGHT)
    if path:
        solutions.append(path)
        print(f"  {i}/{len(waypoints)-1}: {len(path)} steps")
        current = nxt
    else:
        print(f"  {i}/{len(waypoints)-1}: no path found, skipping")

total = sum(len(s) for s in solutions)
print(f"Solved {len(solutions)}/{len(waypoints)-1} segments, {total} frames total.")

ArmAnimator2D(arm).animate_solutions(solutions)

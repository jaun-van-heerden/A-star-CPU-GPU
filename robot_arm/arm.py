import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

STEP_INT = 2
DEG_STEP = 360 // STEP_INT
DEGREES_TO_RADIANS = np.pi / 180


def ccw(A, B, C):
    return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)


# Check if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def closest_point_to_segment(point, segment, threshold=0.5):

    # Define the vectors v and w
    v = segment[1] - segment[0]
    w = point - segment[0]


    if v.real == 0 and v.imag == 0:
        closest = segment[0]  # Or segment[1] as they are the same in this case
    else:
        # Compute the projection of w onto v
        projection = (w.real * v.real + w.imag * v.imag) / (v.real**2 + v.imag**2)

        # Clamp the projection to the [0, 1] range
        projection = max(0, min(1, projection))

        # Compute the closest point using the projection
        closest = segment[0] + projection * v

    # If a threshold is provided, check the distance
    if threshold is not None:
        distance = abs(point - closest)
        if distance < threshold:
            return True

    return False


def select_random_configs(c_space, val, count=1):
    indices = np.argwhere(c_space == val)
    if not indices.size:
        return None
    return [tuple(indices[np.random.choice(len(indices))]) for _ in range(count)]


def segment_segment_distance_3d(p1, p2, p3, p4):
    """Minimum Euclidean distance between 3D line segments P1P2 and P3P4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    if a < 1e-10 and e < 1e-10:
        return float(np.linalg.norm(r))
    if a < 1e-10:
        s, t = 0.0, float(np.clip(f / e, 0, 1))
    else:
        c = np.dot(d1, r)
        if e < 1e-10:
            t, s = 0.0, float(np.clip(-c / a, 0, 1))
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0, 1)) if abs(denom) > 1e-10 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, float(np.clip(-c / a, 0, 1))
            elif t > 1:
                t, s = 1.0, float(np.clip((b - c) / a, 0, 1))

    return float(np.linalg.norm((p1 + s * d1) - (p3 + t * d2)))


class ArmConfiguration:

    def __init__(self, arm_config, obstacle_config) -> None:

        self.arm_config = arm_config
        self.obstacle_config = obstacle_config


    def _validate_config(self, indices_chunk):
        results_chunk = []
        for idx in indices_chunk:
            value = int(not self.self_intersect(idx)) # Your current logic here
            results_chunk.append((idx, value))
        return results_chunk


    def calculate_segments(self, config):

        start_point = complex(0, 0)
        segments = []

        parent_angle = 0
        for angle, arm in zip(config, self.arm_config):
            angle_degrees = STEP_INT * angle
            angle_radians = parent_angle + (angle_degrees * DEGREES_TO_RADIANS)
            end_point = start_point + arm['length'] * np.exp(1j * angle_radians)
            parent_angle = angle_radians - np.pi
            segments.append((start_point, end_point))
            start_point = end_point

        return segments


    def calculate_valid_space(self):
        # This method should calculate and return the valid configuration space (c_space) for the given arm configuration.
        # This could involve iterating through all possible configurations, validating them, and then marking them as valid/invalid.

        num_arms = len(self.arm_config)

        # create c-space grid
        c_space = np.ones((DEG_STEP,) * num_arms, dtype=int)

        for arm_idx, arm in enumerate(self.arm_config):
            angle_limit = arm['angle-limit']
            min_angle = round((angle_limit / 360) * DEG_STEP)
            max_angle = DEG_STEP - min_angle
            slices = [slice(None)] * num_arms
            slices[arm_idx] = slice(0, min_angle)
            c_space[tuple(slices)] = 0
            slices[arm_idx] = slice(max_angle, None)
            c_space[tuple(slices)] = 0

        # Get indices where c_space is 1
        indices = np.argwhere(c_space == 1)

        n_workers = cpu_count()
        indices_chunks = np.array_split(indices, n_workers)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results_chunks = list(executor.map(self._validate_config, indices_chunks))

        # Flatten the results and populate the c_space
        for results in results_chunks:
            for idx, value in results:
                c_space[tuple(idx)] = value

        return c_space


    def plot_arm_configuration(self, config):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        for segment in self.calculate_segments(config):
            real_parts = [c.real for c in segment]
            imag_parts = [c.imag for c in segment]
            ax.plot(real_parts, imag_parts, 'o-')

        max_arm_length = sum([arm['length'] for arm in self.arm_config])
        ax.set_xlim(-max_arm_length, max_arm_length)
        ax.set_ylim(-max_arm_length, max_arm_length)
        ax.grid(True)
        ax.set_aspect('equal', 'box')
        plt.show()


    def intersects_obstacle(self, segment):

        for obstacle in self.obstacle_config:
            if intersect(segment[0], segment[1], obstacle[0], obstacle[1]):
                return True

            for point in segment:
                if closest_point_to_segment(point, obstacle):
                    return True

        return False


    def self_intersect(self, config):

        segments = self.calculate_segments(config)

        # Check if any two segments intersect
        for i in range(len(segments) - 1):  # no need to check the last segment against others
            for j in range(i + 2, len(segments)):  # Start from i+2 to skip the next consecutive segment
                if intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                    return True

        # Check if any segment intersects with obstacles
        for seg in segments[1:]:   # we dont need to check the first one
            if self.intersects_obstacle(seg):
                return True

        return False


    def validate(self, config):
        if self.self_intersect(config):
            raise ValueError("Arm configuration has self-intersections")

    def make_solver(self, deg_step: int):
        """Return a LazyAStarSolver whose valid_fn checks bounds and arm collision."""
        from .solver import LazyAStarSolver
        min_a = [round((a['angle-limit'] / 360) * deg_step) for a in self.arm_config]
        max_a = [deg_step - m for m in min_a]

        def valid_fn(pos):
            for idx, lo, hi in zip(pos, min_a, max_a):
                if not (lo <= idx < hi):
                    return False
            return not self.self_intersect(pos)

        return LazyAStarSolver(
            ndim=len(self.arm_config),
            valid_fn=valid_fn,
            deg_step=deg_step,
        )


class ArmConfiguration3D:
    """
    3-D robot arm. Each joint has two angle indices (azimuth θ, elevation φ).
    Config tuple length = 2 * n_links: (θ0, φ0, θ1, φ1, ...).
    Obstacles are pairs of 3-D numpy arrays: (np.array([x,y,z]), np.array([x,y,z])).
    """

    SELF_INTERSECT_THRESHOLD = 0.05
    OBSTACLE_THRESHOLD = 0.5

    def __init__(self, arm_config: list, obstacle_config: list) -> None:
        self.arm_config = arm_config
        self.obstacle_config = obstacle_config

    def calculate_segments_3d(self, config: tuple) -> list:
        """
        Forward kinematics: config = (θ0_idx, φ0_idx, θ1_idx, φ1_idx, ...).
        Returns list of (start, end) pairs, each a numpy array of shape (3,).
        Uses cumulative spherical coordinates with the same parent-angle
        convention as the 2-D arm (parent_angle -= π at each joint).
        """
        start = np.zeros(3)
        segments = []
        parent_θ = 0.0
        parent_φ = 0.0

        for i, arm in enumerate(self.arm_config):
            θ = parent_θ + STEP_INT * config[2 * i] * DEGREES_TO_RADIANS
            φ = parent_φ + STEP_INT * config[2 * i + 1] * DEGREES_TO_RADIANS
            direction = np.array([
                np.cos(φ) * np.cos(θ),
                np.cos(φ) * np.sin(θ),
                np.sin(φ),
            ])
            end = start + arm['length'] * direction
            segments.append((start.copy(), end.copy()))
            parent_θ = θ - np.pi
            parent_φ = φ - np.pi
            start = end

        return segments

    def intersects_obstacle_3d(self, segment: tuple) -> bool:
        p1, p2 = np.asarray(segment[0], float), np.asarray(segment[1], float)
        for obs in self.obstacle_config:
            if segment_segment_distance_3d(p1, p2, np.asarray(obs[0], float), np.asarray(obs[1], float)) < self.OBSTACLE_THRESHOLD:
                return True
        return False

    def self_intersect_3d(self, config: tuple) -> bool:
        segs = self.calculate_segments_3d(config)
        for i in range(len(segs) - 1):
            for j in range(i + 2, len(segs)):
                if segment_segment_distance_3d(
                    segs[i][0], segs[i][1], segs[j][0], segs[j][1]
                ) < self.SELF_INTERSECT_THRESHOLD:
                    return True
        for seg in segs[1:]:
            if self.intersects_obstacle_3d(seg):
                return True
        return False

    def make_solver(self, deg_step: int):
        """Return a LazyAStarSolver over the 2*n_links dimensional joint space."""
        from .solver import LazyAStarSolver
        n = len(self.arm_config)
        min_a = [round((a['angle-limit'] / 360) * deg_step) for a in self.arm_config]
        max_a = [deg_step - m for m in min_a]

        def valid_fn(pos):
            for i in range(n):
                lo, hi = min_a[i], max_a[i]
                if not (lo <= pos[2 * i] < hi) or not (lo <= pos[2 * i + 1] < hi):
                    return False
            return not self.self_intersect_3d(pos)

        return LazyAStarSolver(ndim=2 * n, valid_fn=valid_fn, deg_step=deg_step)

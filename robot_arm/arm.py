import cmath
import math
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

STEP_INT = 2
DEG_STEP = 360 // STEP_INT
DEGREES_TO_RADIANS = math.pi / 180
_STEP_RAD = STEP_INT * DEGREES_TO_RADIANS
_PI = math.pi


def ccw(A, B, C):
    return (C.imag - A.imag) * (B.real - A.real) > (B.imag - A.imag) * (C.real - A.real)


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


_OBSTACLE_THRESHOLD = 0.15  # minimum clearance between arm segments and obstacle walls


def _dist_pt_seg_2d(pt: complex, a: complex, b: complex) -> float:
    """Minimum distance from complex point pt to segment a–b."""
    v = b - a
    denom = v.real**2 + v.imag**2
    if denom < 1e-12:
        return abs(pt - a)
    t = max(0.0, min(1.0, ((pt - a).real * v.real + (pt - a).imag * v.imag) / denom))
    return abs(pt - (a + t * v))


def closest_point_to_segment(point, segment, threshold=0.5):
    """Legacy proximity helper kept for external callers."""
    return _dist_pt_seg_2d(point, segment[0], segment[1]) < threshold


def select_random_configs(c_space, val, count=1):
    indices = np.argwhere(c_space == val)
    if not indices.size:
        return None
    return [tuple(indices[np.random.choice(len(indices))]) for _ in range(count)]


def segment_segment_distance_3d(p1, p2, p3, p4):
    """Minimum Euclidean distance between 3D line segments P1P2 and P3P4."""
    d1x, d1y, d1z = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    d2x, d2y, d2z = p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]
    rx,  ry,  rz  = p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2]
    a = d1x*d1x + d1y*d1y + d1z*d1z
    e = d2x*d2x + d2y*d2y + d2z*d2z
    f = d2x*rx  + d2y*ry  + d2z*rz

    if a < 1e-10 and e < 1e-10:
        return math.sqrt(rx*rx + ry*ry + rz*rz)
    if a < 1e-10:
        s, t = 0.0, max(0.0, min(1.0, f / e))
    else:
        c = d1x*rx + d1y*ry + d1z*rz
        if e < 1e-10:
            t, s = 0.0, max(0.0, min(1.0, -c / a))
        else:
            b = d1x*d2x + d1y*d2y + d1z*d2z
            denom = a * e - b * b
            s = max(0.0, min(1.0, (b*f - c*e) / denom)) if abs(denom) > 1e-10 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, max(0.0, min(1.0, (b - c) / a))

    qx = p1[0] + s*d1x - p3[0] - t*d2x
    qy = p1[1] + s*d1y - p3[1] - t*d2y
    qz = p1[2] + s*d1z - p3[2] - t*d2z
    return math.sqrt(qx*qx + qy*qy + qz*qz)


class ArmConfiguration:

    def __init__(self, arm_config, obstacle_config) -> None:
        self.arm_config = arm_config
        self.obstacle_config = obstacle_config
        self._lengths = [a['length'] for a in arm_config]

    def _validate_config(self, indices_chunk):
        results_chunk = []
        for idx in indices_chunk:
            value = int(not self.self_intersect(idx))
            results_chunk.append((idx, value))
        return results_chunk

    def calculate_segments(self, config):
        start_point = 0j
        segments = []
        parent_angle = 0.0
        for angle_idx, length in zip(config, self._lengths):
            angle_radians = parent_angle + _STEP_RAD * angle_idx
            end_point = start_point + length * cmath.rect(1.0, angle_radians)
            parent_angle = angle_radians - _PI
            segments.append((start_point, end_point))
            start_point = end_point
        return segments

    def calculate_valid_space(self):
        num_arms = len(self.arm_config)
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

        indices = np.argwhere(c_space == 1)
        n_workers = cpu_count()
        indices_chunks = np.array_split(indices, n_workers)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results_chunks = list(executor.map(self._validate_config, indices_chunks))

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
        max_arm_length = sum(self._lengths)
        ax.set_xlim(-max_arm_length, max_arm_length)
        ax.set_ylim(-max_arm_length, max_arm_length)
        ax.grid(True)
        ax.set_aspect('equal', 'box')
        plt.show()

    def intersects_obstacle(self, segment):
        p0, p1 = segment
        for obs in self.obstacle_config:
            if isinstance(obs[1], (int, float)):
                # Circle obstacle: (center, radius)
                center, radius = obs
                v = p1 - p0
                denom = v.real**2 + v.imag**2
                t = (max(0.0, min(1.0, ((center - p0).real * v.real + (center - p0).imag * v.imag) / denom))
                     if denom > 1e-12 else 0.0)
                if abs(center - (p0 + t * v)) < radius:
                    return True
            else:
                # Line segment obstacle — proper crossing OR within threshold.
                # Check all four endpoint-to-segment distances so the arm can
                # never graze an obstacle tip or have the obstacle cross its middle.
                q0, q1 = obs[0], obs[1]
                if intersect(p0, p1, q0, q1):
                    return True
                if (_dist_pt_seg_2d(p0, q0, q1) < _OBSTACLE_THRESHOLD or
                        _dist_pt_seg_2d(p1, q0, q1) < _OBSTACLE_THRESHOLD or
                        _dist_pt_seg_2d(q0, p0, p1) < _OBSTACLE_THRESHOLD or
                        _dist_pt_seg_2d(q1, p0, p1) < _OBSTACLE_THRESHOLD):
                    return True
        return False

    def self_intersect(self, config):
        segments = self.calculate_segments(config)
        for i in range(len(segments) - 1):
            for j in range(i + 2, len(segments)):
                if intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                    return True
        for seg in segments:
            if self.intersects_obstacle(seg):
                return True
        return False

    def random_valid_config(self, deg_step: int, max_tries: int = 2000):
        import random
        min_a = [round((a['angle-limit'] / 360) * deg_step) for a in self.arm_config]
        max_a = [deg_step - m for m in min_a]
        for _ in range(max_tries):
            cfg = tuple(random.randint(lo, hi - 1) for lo, hi in zip(min_a, max_a))
            if not self.self_intersect(cfg):
                return cfg
        return None

    def validate(self, config):
        if self.self_intersect(config):
            raise ValueError("Arm configuration has self-intersections")

    def make_solver(self, deg_step: int):
        """Return a LazyAStarSolver whose valid_fn checks bounds and arm collision."""
        from .solver import LazyAStarSolver
        min_a = [round((a['angle-limit'] / 360) * deg_step) for a in self.arm_config]
        max_a = [deg_step - m for m in min_a]
        _cache: dict = {}

        def valid_fn(pos):
            cached = _cache.get(pos)
            if cached is not None:
                return cached
            for idx, lo, hi in zip(pos, min_a, max_a):
                if not (lo <= idx < hi):
                    _cache[pos] = False
                    return False
            result = not self.self_intersect(pos)
            _cache[pos] = result
            return result

        return LazyAStarSolver(ndim=len(self.arm_config), valid_fn=valid_fn, deg_step=deg_step)


class ArmConfiguration3D:
    """
    3-D robot arm. Each joint has two angle indices (azimuth θ, elevation φ).
    Config tuple length = 2 * n_links: (θ0, φ0, θ1, φ1, ...).
    Obstacles are pairs of 3-element sequences: (array/list [x,y,z], array/list [x,y,z]).
    """

    SELF_INTERSECT_THRESHOLD = 0.05
    OBSTACLE_THRESHOLD = 0.5

    def __init__(self, arm_config: list, obstacle_config: list) -> None:
        self.arm_config = arm_config
        self.obstacle_config = obstacle_config
        self._lengths = [a['length'] for a in arm_config]

    def calculate_segments_3d(self, config: tuple) -> list:
        """
        Forward kinematics: config = (θ0_idx, φ0_idx, θ1_idx, φ1_idx, ...).
        Returns list of (start, end) pairs as plain [x, y, z] lists.
        Uses cumulative spherical coordinates with the same parent-angle
        convention as the 2-D arm (parent_angle -= π at each joint).
        """
        start = [0.0, 0.0, 0.0]
        segments = []
        parent_θ = 0.0
        parent_φ = 0.0

        for i, length in enumerate(self._lengths):
            θ = parent_θ + _STEP_RAD * config[2 * i]
            φ = parent_φ + _STEP_RAD * config[2 * i + 1]
            cosφ = math.cos(φ)
            end = [
                start[0] + length * cosφ * math.cos(θ),
                start[1] + length * cosφ * math.sin(θ),
                start[2] + length * math.sin(φ),
            ]
            segments.append((start[:], end[:]))
            parent_θ = θ - _PI
            parent_φ = φ - _PI
            start = end

        return segments

    def intersects_obstacle_3d(self, segment: tuple) -> bool:
        p1, p2 = segment[0], segment[1]
        for obs in self.obstacle_config:
            if segment_segment_distance_3d(p1, p2, obs[0], obs[1]) < self.OBSTACLE_THRESHOLD:
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
        for seg in segs:
            if self.intersects_obstacle_3d(seg):
                return True
        return False

    def random_valid_config(self, deg_step: int, max_tries: int = 2000):
        import random
        n = len(self.arm_config)
        min_a = [round((a['angle-limit'] / 360) * deg_step) for a in self.arm_config]
        max_a = [deg_step - m for m in min_a]
        for _ in range(max_tries):
            parts = []
            for i in range(n):
                parts.append(random.randint(min_a[i], max_a[i] - 1))
                parts.append(random.randint(min_a[i], max_a[i] - 1))
            cfg = tuple(parts)
            if not self.self_intersect_3d(cfg):
                return cfg
        return None

    def make_solver(self, deg_step: int):
        """Return a LazyAStarSolver over the 2*n_links dimensional joint space."""
        from .solver import LazyAStarSolver
        n = len(self.arm_config)
        min_a = [round((a['angle-limit'] / 360) * deg_step) for a in self.arm_config]
        max_a = [deg_step - m for m in min_a]
        _cache: dict = {}

        def valid_fn(pos):
            cached = _cache.get(pos)
            if cached is not None:
                return cached
            for i in range(n):
                lo, hi = min_a[i], max_a[i]
                if not (lo <= pos[2 * i] < hi) or not (lo <= pos[2 * i + 1] < hi):
                    _cache[pos] = False
                    return False
            result = not self.self_intersect_3d(pos)
            _cache[pos] = result
            return result

        return LazyAStarSolver(ndim=2 * n, valid_fn=valid_fn, deg_step=deg_step)

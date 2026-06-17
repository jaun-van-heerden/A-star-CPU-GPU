"""
Tests for robot_arm package: solver, 2D arm kinematics/collision, 3D arm kinematics.
No display dependencies — pure logic tests.
"""
import math
import numpy as np
import pytest

from robot_arm.solver import LazyAStarSolver
from robot_arm.arm import ArmConfiguration, ArmConfiguration3D, DEG_STEP, STEP_INT


# ── helpers ──────────────────────────────────────────────────────────────────

def _open_valid(ndim, size=200):
    """valid_fn: open grid of given size per axis."""
    def fn(pos):
        return all(0 <= x < size for x in pos)
    return fn


def _arm2d(n_links=3, length=1, angle_limit=10, obstacles=()):
    return ArmConfiguration(
        [{'name': f'arm{i:02d}', 'length': length, 'angle-limit': angle_limit}
         for i in range(1, n_links + 1)],
        obstacle_config=list(obstacles),
    )


def _arm3d(n_links=3, length=1, angle_limit=10, obstacles=()):
    return ArmConfiguration3D(
        [{'name': f'arm{i:02d}', 'length': length, 'angle-limit': angle_limit}
         for i in range(1, n_links + 1)],
        obstacle_config=list(obstacles),
    )


# ── LazyAStarSolver ───────────────────────────────────────────────────────────

class TestSolver:

    def test_trivial_adjacent(self):
        s = LazyAStarSolver(1, _open_valid(1), 100)
        assert s.solve((0,), (1,)) == [(0,), (1,)]

    def test_same_start_goal(self):
        s = LazyAStarSolver(1, _open_valid(1), 100)
        assert s.solve((5,), (5,)) == [(5,)]

    def test_invalid_start_returns_none(self):
        def fn(pos): return pos[0] != 0
        s = LazyAStarSolver(1, fn, 10)
        assert s.solve((0,), (5,)) is None

    def test_invalid_goal_returns_none(self):
        def fn(pos): return pos[0] != 9
        s = LazyAStarSolver(1, fn, 10)
        assert s.solve((0,), (9,)) is None

    def test_unreachable_goal_returns_none(self):
        # 6 is surrounded by walls at 5 and 7 in 1-D
        walls = {(5,), (7,)}
        def fn(pos): return pos not in walls and 0 <= pos[0] < 10
        s = LazyAStarSolver(1, fn, 10)
        assert s.solve((4,), (6,)) is None

    def test_path_starts_and_ends_correctly(self):
        s = LazyAStarSolver(2, _open_valid(2), 20)
        path = s.solve((0, 0), (7, 3))
        assert path[0] == (0, 0)
        assert path[-1] == (7, 3)

    def test_chebyshev_optimal_length(self):
        """With diagonal moves costing 1, path length == Chebyshev distance + 1."""
        s = LazyAStarSolver(2, _open_valid(2), 50)
        path = s.solve((0, 0), (3, 7))
        assert path is not None
        # Chebyshev(0,0→3,7) = max(3,7) = 7 steps → 8 nodes
        assert len(path) == 8

    def test_path_continuity(self):
        """Each consecutive pair of configs in the path must be neighbours."""
        s = LazyAStarSolver(2, _open_valid(2), 50)
        path = s.solve((1, 2), (10, 8))
        assert path is not None
        for a, b in zip(path, path[1:]):
            assert max(abs(x - y) for x, y in zip(a, b)) == 1, \
                f"Non-adjacent step: {a} → {b}"

    def test_3d_open_grid(self):
        s = LazyAStarSolver(3, _open_valid(3, 30), 30)
        path = s.solve((0, 0, 0), (2, 4, 1))
        assert path is not None
        assert path[0] == (0, 0, 0)
        assert path[-1] == (2, 4, 1)


# ── ArmConfiguration 2D kinematics ───────────────────────────────────────────

class TestArmKinematics2D:

    def test_single_joint_up(self):
        """Index 45 → 90° → arm points up."""
        arm = _arm2d(n_links=1)
        segs = arm.calculate_segments((45,))
        assert len(segs) == 1
        _, end = segs[0]
        assert abs(end.real) < 1e-9
        assert abs(end.imag - 1.0) < 1e-9

    def test_single_joint_left(self):
        """Index 90 → 180° → arm points left."""
        arm = _arm2d(n_links=1)
        _, end = arm.calculate_segments((90,))[0]
        assert abs(end.real - (-1.0)) < 1e-9
        assert abs(end.imag) < 1e-9

    def test_two_joint_elbow(self):
        """Joint0=45(up), joint1=45 relative → arm goes up then right."""
        arm = _arm2d(n_links=2)
        segs = arm.calculate_segments((45, 45))
        # seg0: (0,0)→(0,1)
        assert abs(segs[0][1].real) < 1e-9
        assert abs(segs[0][1].imag - 1.0) < 1e-9
        # seg1: (0,1)→(1,1) (right)
        assert abs(segs[1][1].real - 1.0) < 1e-9
        assert abs(segs[1][1].imag - 1.0) < 1e-9

    def test_segment_count(self):
        arm = _arm2d(n_links=4)
        assert len(arm.calculate_segments((45, 45, 45, 45))) == 4

    def test_segment_continuity(self):
        """End of segment i == start of segment i+1."""
        arm = _arm2d(n_links=3)
        segs = arm.calculate_segments((30, 60, 90))
        for i in range(len(segs) - 1):
            assert abs(segs[i][1] - segs[i + 1][0]) < 1e-12


# ── ArmConfiguration 2D collision ────────────────────────────────────────────

class TestArmCollision2D:

    def test_no_collision_straight(self):
        """Straight arm doesn't self-intersect."""
        arm = _arm2d(n_links=3)
        assert not arm.self_intersect((90, 90, 90))

    def test_collision_with_demo_obstacle(self):
        """Config (32,62,21) has seg1 crossing the obstacle y=1 at x≈1.16."""
        arm = _arm2d(n_links=3, obstacles=[(complex(1, 1), complex(3, 1))])
        assert arm.self_intersect((32, 62, 21))

    def test_no_collision_arms_up(self):
        """Arms extending upward (45,90,45) stay clear of obstacles placed right/low."""
        arm = _arm2d(n_links=3, obstacles=[
            (complex(-2, 1), complex(-2, 0)),
            (complex(1, 1), complex(3, 1)),
        ])
        # Arm traces: (0,0)→(0,1)→(0,2)→(1,2) — all above y=1 or at x=0
        assert not arm.self_intersect((45, 90, 45))

    def test_obstacle_collision(self):
        """Arm segment hits a vertical obstacle line."""
        # joint0=45→up, joint1=45→right → seg1 = (0,1)→(1,1)
        # Obstacle: x=0.5 from y=0.5 to y=1.5 — crosses seg1 at (0.5, 1)
        arm = _arm2d(
            n_links=2,
            obstacles=[(complex(0.5, 0.5), complex(0.5, 1.5))],
        )
        assert arm.self_intersect((45, 45))

    def test_no_obstacle_collision_when_clear(self):
        """Same arm, obstacle moved out of path."""
        arm = _arm2d(
            n_links=2,
            obstacles=[(complex(2.0, 0.5), complex(2.0, 1.5))],  # x=2, well past arm
        )
        assert not arm.self_intersect((45, 45))


# ── ArmConfiguration3D kinematics ────────────────────────────────────────────

class TestArmKinematics3D:

    def test_single_joint_yaw_right(self):
        """θ=0°, φ=0° → arm points in +X direction."""
        arm = _arm3d(n_links=1)
        segs = arm.calculate_segments_3d((0, 0))
        _, end = segs[0]
        assert np.allclose(end, [1, 0, 0], atol=1e-9)

    def test_single_joint_yaw_up(self):
        """θ=90°(idx=45), φ=0° → arm points in +Y direction."""
        arm = _arm3d(n_links=1)
        _, end = arm.calculate_segments_3d((45, 0))[0]
        assert np.allclose(end, [0, 1, 0], atol=1e-9)

    def test_single_joint_elevation(self):
        """θ=0°, φ=90°(idx=45) → arm points in +Z direction."""
        arm = _arm3d(n_links=1)
        _, end = arm.calculate_segments_3d((0, 45))[0]
        assert np.allclose(end, [0, 0, 1], atol=1e-9)

    def test_segment_count(self):
        arm = _arm3d(n_links=3)
        assert len(arm.calculate_segments_3d((45, 0, 45, 0, 45, 0))) == 3

    def test_segment_continuity(self):
        """End of segment i == start of segment i+1."""
        arm = _arm3d(n_links=3)
        segs = arm.calculate_segments_3d((30, 10, 60, 20, 45, 5))
        for i in range(len(segs) - 1):
            assert np.allclose(segs[i][1], segs[i + 1][0], atol=1e-12)


# ── ArmConfiguration3D collision ─────────────────────────────────────────────

class TestArmCollision3D:

    def test_no_collision_extended(self):
        """Arm pointing in consistent direction should not self-intersect."""
        arm = _arm3d(n_links=3)
        assert not arm.self_intersect_3d((45, 0, 45, 0, 45, 0))

    def test_obstacle_collision_3d(self):
        """Arm hits a 3D obstacle segment."""
        # joint0: θ=45(→90°), φ=0 → seg0 from (0,0,0) to (0,1,0)
        # joint1: same → seg1 from (0,1,0) to (0,2,0)  [upward extension]
        # Actually let's compute: parent_θ after joint0 = π/2-π = -π/2
        #                         parent_φ after joint0 = 0-π = -π
        # joint1: θ = -π/2 + 45*2°*π/180 = -π/2+π/2 = 0
        #         φ = -π + 0*2°*π/180 = -π
        # direction1 = (cos(-π)*cos(0), cos(-π)*sin(0), sin(-π)) = (-1, 0, 0)
        # So seg1 goes from (0,1,0) to (-1,1,0)
        # Put obstacle crossing seg1: from (-0.5, 1, -0.5) to (-0.5, 1, 0.5)
        arm = _arm3d(
            n_links=2,
            obstacles=[(np.array([-0.5, 1, -0.5]), np.array([-0.5, 1, 0.5]))],
        )
        assert arm.self_intersect_3d((45, 0, 45, 0))

    def test_no_obstacle_collision_3d_when_clear(self):
        """Same arm config, obstacle placed well away from arm path."""
        arm = _arm3d(
            n_links=2,
            obstacles=[(np.array([5.0, 5.0, 5.0]), np.array([6.0, 5.0, 5.0]))],
        )
        assert not arm.self_intersect_3d((45, 0, 45, 0))


# ── make_solver integration ───────────────────────────────────────────────────

class TestMakeSolver:

    def test_2d_one_step_path(self):
        """Adjacent valid configs → path of length 2."""
        arm = _arm2d(n_links=3)
        solver = arm.make_solver(DEG_STEP)
        path = solver.solve((32, 62, 21), (33, 62, 21))
        assert path == [(32, 62, 21), (33, 62, 21)]

    def test_2d_invalid_config_blocked(self):
        """Index below min_angle → solver returns None."""
        arm = _arm2d(n_links=3)
        solver = arm.make_solver(DEG_STEP)
        # min_angle = round(10/360 * 180) = 5; index 4 is invalid
        assert solver.solve((4, 62, 21), (33, 62, 21)) is None

    def test_2d_longer_path(self):
        """Solve a multi-step path, verify it's valid and continuous."""
        arm = _arm2d(n_links=3)
        solver = arm.make_solver(DEG_STEP)
        path = solver.solve((32, 62, 21), (40, 55, 30))
        assert path is not None
        assert path[0] == (32, 62, 21)
        assert path[-1] == (40, 55, 30)
        # Chebyshev optimal: max(|40-32|, |55-62|, |30-21|) = max(8,7,9) = 9
        assert len(path) == 10

    def test_3d_one_step_path(self):
        """Adjacent valid 3D configs → path of length 2."""
        arm = _arm3d(n_links=2)
        solver = arm.make_solver(DEG_STEP)
        path = solver.solve((32, 45, 62, 50), (33, 45, 62, 50))
        assert path == [(32, 45, 62, 50), (33, 45, 62, 50)]

    def test_3d_invalid_config_blocked(self):
        """Index below min_angle in 3D → None."""
        arm = _arm3d(n_links=2)
        solver = arm.make_solver(DEG_STEP)
        assert solver.solve((4, 45, 62, 50), (33, 45, 62, 50)) is None

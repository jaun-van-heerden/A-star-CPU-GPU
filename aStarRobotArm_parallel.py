import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from aStarXd import AStarSolver

from robot_arm import ArmConfiguration, ArmAnimator2D, DEG_STEP, STEP_INT
from robot_arm.arm import select_random_configs


def plot_segments(segments):
    for seg in segments:
        plt.plot([seg[0].real, seg[1].real], [seg[0].imag, seg[1].imag], 'o-')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Visualization of Complex Number Segments')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.2)
    plt.axvline(0, color='black', linewidth=0.2)
    plt.show()


def visualize_c_space_slice(c_space, joint1, joint2):
    c_space_slice = c_space.sum(axis=tuple(i for i in range(c_space.ndim) if i not in [joint1, joint2]))
    plt.imshow(c_space_slice, cmap='gray_r', interpolation='none', origin='lower')
    plt.colorbar(label='Number of valid configurations')
    plt.title(f"C-Space for Joint {joint1 + 1} and Joint {joint2 + 1}")
    plt.xlabel(f"Joint {joint1 + 1} angle (increments of {STEP_INT} degrees)")
    plt.ylabel(f"Joint {joint2 + 1} angle (increments of {STEP_INT} degrees)")
    plt.show()


def visualize_all_c_space_slices(c_space):
    num_joints = c_space.ndim
    joint_combinations = list(combinations(range(num_joints), 2))
    num_combinations = len(joint_combinations)

    nrows = num_combinations // 2 if num_combinations % 2 == 0 else num_combinations // 2 + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 6 * nrows))
    if num_combinations % 2 != 0:
        axes[-1, -1].axis('off')

    for idx, (joint1, joint2) in enumerate(joint_combinations):
        ax = axes[idx // 2, idx % 2] if num_combinations > 2 else axes[idx]
        c_space_slice = c_space.sum(axis=tuple(i for i in range(num_joints) if i not in [joint1, joint2]))
        cax = ax.imshow(c_space_slice, cmap='gray_r', interpolation='none', origin='lower')
        fig.colorbar(cax, ax=ax, label='Number of valid configurations')
        ax.set_title(f"C-Space for Joint {joint1 + 1} and Joint {joint2 + 1}")
        ax.set_xlabel(f"Joint {joint1 + 1} angle (increments of {STEP_INT} degrees)")
        ax.set_ylabel(f"Joint {joint2 + 1} angle (increments of {STEP_INT} degrees)")

    plt.tight_layout()
    plt.show()


def visualize_c_space_slice_path(c_space, joint1, joint2, solution_path=None):
    c_space_slice = c_space.sum(axis=tuple(i for i in range(c_space.ndim) if i not in [joint1, joint2]))
    plt.imshow(c_space_slice, cmap='gray_r', interpolation='none', origin='lower')
    plt.colorbar(label='Number of valid configurations')
    plt.title(f"C-Space for Joint {joint1 + 1} and Joint {joint2 + 1}")
    if solution_path is not None:
        plt.plot(
            [cfg[joint1] for cfg in solution_path],
            [cfg[joint2] for cfg in solution_path],
            color='red', marker='o', linewidth=2, markersize=4,
        )
    plt.show()


if __name__ == "__main__":
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
    all_solutions = []
    current_config = waypoints[0]
    for next_config in waypoints[1:]:
        solution = solver.solve(current_config, next_config)
        if solution:
            all_solutions.append(solution)
            current_config = next_config
        else:
            print(f"No path: {current_config} → {next_config}")

    animator.animate_solutions(all_solutions)

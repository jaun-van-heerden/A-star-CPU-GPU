import heapq
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from aStarXd import AStarSolver
from itertools import combinations, product

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

STEP_INT = 2
DEG_STEP = 360 // STEP_INT

DEGREES_TO_RADIANS = np.pi / 180


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def plot_segments(segments):
    """
    Visualize a list of segments represented as complex numbers on a 2D plane.

    Parameters:
    - segments (list): A list of tuples. Each tuple contains two complex numbers representing the start and end of a segment.
    """
    # Plot each segment
    for seg in segments:
        plt.plot([seg[0].real, seg[1].real], [seg[0].imag, seg[1].imag], 'o-')

    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Visualization of Complex Number Segments')
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.2)
    plt.axvline(0, color='black',linewidth=0.2)
    plt.show()



def select_random_configs(c_space, val, count=1):
    indices = np.argwhere(c_space == val)
    if not indices.size:
        return None
    return [tuple(indices[np.random.choice(len(indices))]) for _ in range(count)]




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





class ArmAnimator:

    def __init__(self, arm):
        self.arm = arm

    def animate_solutions(self, solutions):
        if not solutions:
            print("No solutions to animate.")
            return

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        flat = [cfg for sol in solutions for cfg in sol]

        # precompute joint positions for every frame up front
        # each frame: list of (x, y) from base through every joint to end-effector
        frames = []
        for cfg in flat:
            xs, ys = [0.0], [0.0]
            for seg in self.arm.calculate_segments(cfg):
                xs.append(seg[1].real)
                ys.append(seg[1].imag)
            frames.append((xs, ys))

        # per-frame target: reuse end-effector from already-computed frames
        targets = []
        offset = 0
        for sol in solutions:
            xs, ys = frames[offset + len(sol) - 1]
            targets.extend([(xs[-1], ys[-1])] * len(sol))
            offset += len(sol)

        total_frames = len(flat)
        total_len = sum(a['length'] for a in self.arm.arm_config)

        # --- window layout ---
        win = QtWidgets.QMainWindow()
        win.setWindowTitle("Robotic Arm — A* Path")
        central = QtWidgets.QWidget()
        win.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # --- plot ---
        plot = pg.PlotWidget()
        plot.setAspectLocked(True)
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel('bottom', 'x')
        plot.setLabel('left', 'y')
        plot.setXRange(-total_len, total_len)
        plot.setYRange(-total_len, total_len)
        vbox.addWidget(plot)

        # obstacles (static, drawn once)
        for obs in self.arm.obstacle_config:
            plot.plot([obs[0].real, obs[1].real], [obs[0].imag, obs[1].imag],
                      pen=pg.mkPen('r', width=3))

        # overall start / end endpoint markers (static)
        plot.plot([frames[0][0][-1]], [frames[0][1][-1]],
                  symbol='o', symbolSize=12, symbolBrush='g', pen=None)
        plot.plot([frames[-1][0][-1]], [frames[-1][1][-1]],
                  symbol='o', symbolSize=12, symbolBrush='r', pen=None)

        # animated arm
        arm_curve = plot.plot(
            [], [],
            pen=pg.mkPen('b', width=2),
            symbol='o', symbolSize=8, symbolBrush='b', symbolPen=None,
        )

        # animated target marker
        target_item = pg.ScatterPlotItem(
            symbol='x', size=16,
            brush=pg.mkBrush('y'), pen=pg.mkPen('y', width=2),
        )
        plot.addItem(target_item)

        # --- controls ---
        ctrl = QtWidgets.QHBoxLayout()

        play_btn = QtWidgets.QPushButton("Pause")
        play_btn.setFixedWidth(70)

        speed_lbl = QtWidgets.QLabel("3×")
        speed_lbl.setFixedWidth(28)

        speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        speed_slider.setRange(1, 10)
        speed_slider.setValue(3)
        speed_slider.setFixedWidth(160)

        frame_lbl = QtWidgets.QLabel(f"1 / {total_frames}")

        ctrl.addWidget(play_btn)
        ctrl.addSpacing(8)
        ctrl.addWidget(QtWidgets.QLabel("Speed:"))
        ctrl.addWidget(speed_slider)
        ctrl.addWidget(speed_lbl)
        ctrl.addStretch()
        ctrl.addWidget(frame_lbl)
        vbox.addLayout(ctrl)

        frame = 0
        playing = True
        BASE_MS = 80

        def tick():
            nonlocal frame, playing
            if frame >= total_frames:
                timer.stop()
                play_btn.setText("Play")
                playing = False
                return
            arm_curve.setData(*frames[frame])
            tx, ty = targets[frame]
            target_item.setData([tx], [ty])
            frame_lbl.setText(f"{frame + 1} / {total_frames}")
            frame += 1

        def toggle():
            nonlocal frame, playing
            playing = not playing
            if playing:
                if frame >= total_frames:
                    frame = 0
                play_btn.setText("Pause")
                timer.start()
            else:
                play_btn.setText("Play")
                timer.stop()

        def set_speed(val):
            speed_lbl.setText(f"{val}×")
            timer.setInterval(max(1, BASE_MS // val))

        play_btn.clicked.connect(toggle)
        speed_slider.valueChanged.connect(set_speed)

        timer = QtCore.QTimer()
        timer.setInterval(BASE_MS // speed_slider.value())
        timer.timeout.connect(tick)
        timer.start()

        win.resize(720, 780)
        win.show()
        app.exec_()



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


class LazyAStarSolver:
    """
    A* over the discrete joint-angle space with on-the-fly collision checking.
    Avoids precomputing the full D^N C-space grid — only evaluates configurations
    that A* actually visits, so it scales much better with additional links.
    Produces optimal paths (shortest step-count in joint space) via an admissible
    Chebyshev (L∞) heuristic, identical to running A* on a precomputed grid.
    """

    def __init__(self, arm, deg_step):
        self.arm = arm
        self.deg_step = deg_step
        self.ndim = len(arm.arm_config)
        self._min = [round((a['angle-limit'] / 360) * deg_step) for a in arm.arm_config]
        self._max = [deg_step - m for m in self._min]
        self._offsets = [o for o in product([-1, 0, 1], repeat=self.ndim) if any(x != 0 for x in o)]

    def _valid(self, pos):
        for idx, lo, hi in zip(pos, self._min, self._max):
            if not (lo <= idx < hi):
                return False
        return not self.arm.self_intersect(pos)

    def _neighbors(self, pos):
        for offset in self._offsets:
            nb = tuple(p + o for p, o in zip(pos, offset))
            if self._valid(nb):
                yield nb

    def _heuristic(self, a, b):
        return max(abs(x - y) for x, y in zip(a, b))

    def solve(self, start, goal):
        if not self._valid(start) or not self._valid(goal):
            return None

        open_list = [(self._heuristic(start, goal), start)]
        came_from = {}
        best_cost = {start: 0}
        visited = set()

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            if current in visited:
                continue
            visited.add(current)

            for nb in self._neighbors(current):
                next_cost = best_cost[current] + 1
                if next_cost < best_cost.get(nb, float('inf')):
                    best_cost[nb] = next_cost
                    came_from[nb] = current
                    heapq.heappush(open_list, (next_cost + self._heuristic(nb, goal), nb))

        return None




def visualize_c_space_slice(c_space, joint1, joint2):
    """
    Visualizes the configuration space for two specified joints.
    """
    # Extracting the 2D slice for the given joints
    c_space_slice = c_space.sum(axis=tuple([i for i in range(c_space.ndim) if i not in [joint1, joint2]]))
    
    # Plotting
    plt.imshow(c_space_slice, cmap='gray_r', interpolation='none', origin='lower')
    plt.colorbar(label='Number of valid configurations')
    plt.title(f"C-Space for Joint {joint1 + 1} and Joint {joint2 + 1}")
    plt.xlabel(f"Joint {joint1 + 1} angle (increments of {STEP_INT} degrees)")
    plt.ylabel(f"Joint {joint2 + 1} angle (increments of {STEP_INT} degrees)")
    plt.show()
    
    
    


def visualize_all_c_space_slices(c_space):
    """
    Visualizes the configuration space for all unique joint combinations using subplots.
    """
    num_joints = c_space.ndim
    joint_combinations = list(combinations(range(num_joints), 2))
    num_combinations = len(joint_combinations)

    # Set up subplots
    fig, axes = plt.subplots(nrows=num_combinations//2 if num_combinations%2 == 0 else (num_combinations//2 + 1),
                             ncols=2, figsize=(12, 6*num_combinations//2))
    if num_combinations % 2 != 0:
        axes[-1, -1].axis('off')  # Turn off the last subplot if the number of combinations is odd

    for idx, (joint1, joint2) in enumerate(joint_combinations):
        ax = axes[idx//2, idx%2] if num_combinations > 2 else axes[idx]

        # Extracting the 2D slice for the given joints
        c_space_slice = c_space.sum(axis=tuple([i for i in range(num_joints) if i not in [joint1, joint2]]))
        
        # Plotting on the specified subplot
        cax = ax.imshow(c_space_slice, cmap='gray_r', interpolation='none', origin='lower')
        fig.colorbar(cax, ax=ax, label='Number of valid configurations')
        ax.set_title(f"C-Space for Joint {joint1 + 1} and Joint {joint2 + 1}")
        ax.set_xlabel(f"Joint {joint1 + 1} angle (increments of {STEP_INT} degrees)")
        ax.set_ylabel(f"Joint {joint2 + 1} angle (increments of {STEP_INT} degrees)")
    
    plt.tight_layout()
    plt.show()



def visualize_c_space_slice_path(c_space, joint1, joint2, solution_path=None):
    c_space_slice = c_space.sum(axis=tuple([i for i in range(c_space.ndim) if i not in [joint1, joint2]]))
    plt.imshow(c_space_slice, cmap='gray_r', interpolation='none', origin='lower')
    plt.colorbar(label='Number of valid configurations')
    plt.title(f"C-Space for Joint {joint1 + 1} and Joint {joint2 + 1}")
    
    if solution_path is not None:
        path_joint1 = [config[joint1] for config in solution_path]
        path_joint2 = [config[joint2] for config in solution_path]
        plt.plot(path_joint1, path_joint2, color='red', marker='o', linewidth=2, markersize=4)
        
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
        (complex(1, 1), complex(3, 1))
    ]

    Arm = ArmConfiguration(arm_config, obstacle_segments)
    solver = LazyAStarSolver(Arm, DEG_STEP)

    random_configs = [
        (32, 62, 21), (33, 26, 51), (30, 18, 34), (60, 45, 58), (40, 51, 65),
        (5, 7, 22), (54, 57, 36), (40, 68, 34), (57, 58, 43), (22, 26, 38)
    ]

    animator = ArmAnimator(Arm)

    all_solutions = []
    current_config = random_configs[0]
    for next_config in random_configs[1:]:
        solution = solver.solve(current_config, next_config)
        if solution:
            all_solutions.append(solution)
            current_config = next_config
        else:
            print(f"No solution found between {current_config} and {next_config}!")

    animator.animate_solutions(all_solutions)
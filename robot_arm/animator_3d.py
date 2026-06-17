import numpy as np


class ArmAnimator3D:
    """Interactive 3-D animation of A* arm paths using pyqtgraph OpenGL."""

    def __init__(self, arm):
        self.arm = arm

    def animate_solutions(self, solutions):
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtWidgets, QtCore

        if not solutions:
            print("No solutions to animate.")
            return

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        flat = [cfg for sol in solutions for cfg in sol]

        # Precompute per-frame joint positions as (n_joints+1, 3) arrays.
        # Row 0 = origin; subsequent rows = each joint's endpoint.
        frames = []
        for cfg in flat:
            pts = [np.zeros(3)]
            for seg in self.arm.calculate_segments_3d(cfg):
                pts.append(seg[1])
            frames.append(np.array(pts, dtype=float))

        # Per-frame target: end-effector of each solution's goal config.
        targets = []
        offset = 0
        for sol in solutions:
            ep = frames[offset + len(sol) - 1][-1]
            targets.extend([ep] * len(sol))
            offset += len(sol)

        total_frames = len(flat)
        total_len = sum(a['length'] for a in self.arm.arm_config)

        # --- window layout ---
        win = QtWidgets.QMainWindow()
        win.setWindowTitle("Robotic Arm 3D — A* Path")
        central = QtWidgets.QWidget()
        win.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # --- 3D GL view ---
        view = gl.GLViewWidget()
        view.setCameraPosition(distance=total_len * 4)
        view.setMinimumHeight(500)
        vbox.addWidget(view)

        # Reference grid on XY plane
        grid = gl.GLGridItem()
        grid.setSize(total_len * 2, total_len * 2)
        grid.setSpacing(total_len / 5, total_len / 5)
        view.addItem(grid)

        # Coordinate axes
        axis = gl.GLAxisItem()
        axis.setSize(total_len, total_len, total_len)
        view.addItem(axis)

        # Obstacles (static red lines)
        for obs in self.arm.obstacle_config:
            pts = np.array([obs[0], obs[1]], dtype=float)
            view.addItem(gl.GLLinePlotItem(pos=pts, color=(1, 0, 0, 1), width=4, antialias=True))

        # Start marker (green sphere-like scatter point)
        start_item = gl.GLScatterPlotItem(
            pos=frames[0][[-1]], color=(0, 1, 0, 1), size=14,
        )
        view.addItem(start_item)

        # End marker (red)
        end_item = gl.GLScatterPlotItem(
            pos=frames[-1][[-1]], color=(1, 0, 0, 1), size=14,
        )
        view.addItem(end_item)

        # Animated arm line (blue)
        arm_line = gl.GLLinePlotItem(
            pos=frames[0], color=(0.2, 0.6, 1, 1), width=3, antialias=True,
        )
        view.addItem(arm_line)

        # Animated target marker (yellow)
        target_item = gl.GLScatterPlotItem(
            pos=targets[0].reshape(1, 3), color=(1, 1, 0, 1), size=12,
        )
        view.addItem(target_item)

        # --- controls ---
        ctrl = QtWidgets.QHBoxLayout()

        play_btn = QtWidgets.QPushButton("Pause")
        play_btn.setFixedWidth(70)

        speed_lbl = QtWidgets.QLabel("3×")
        speed_lbl.setFixedWidth(28)

        speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
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
            arm_line.setData(pos=frames[frame])
            target_item.setData(pos=targets[frame].reshape(1, 3))
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

        win.resize(800, 680)
        win.show()
        app.exec()

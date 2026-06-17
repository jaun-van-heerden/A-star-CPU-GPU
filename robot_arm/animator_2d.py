import numpy as np


class ArmAnimator2D:

    def __init__(self, arm):
        self.arm = arm

    def animate_solutions(self, solutions):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets, QtCore

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
            if isinstance(obs[1], (int, float)):
                # Circle obstacle — draw outline
                theta = np.linspace(0, 2 * np.pi, 128)
                cx, cy, r = obs[0].real, obs[0].imag, float(obs[1])
                plot.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                          pen=pg.mkPen('r', width=2))
            else:
                # Line segment obstacle
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
        app.exec()

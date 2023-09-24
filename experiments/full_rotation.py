import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



DEG_INT = 5
DEG_STEP = 360 // DEG_INT


obstacle_segments = [
    (complex(1, 1), complex(3, 1)), 
    (complex(1, 2), complex(2, 1))
    ]


arm_config = [
    {'name': 'arm01', 'length': 1, 'angle-limit': 5},
    {'name': 'arm02', 'length': 1, 'angle-limit': 5},
    {'name': 'arm03', 'length': 1, 'angle-limit': 5}
]



def _config_to_xy(config):
    total_angle = 0
    xdata, ydata = [0], [0]
    for angle, arm in zip(config, arm_config):
        angle_degrees = DEG_INT * angle
        
        angle_normalized = angle_degrees  - 180  # Ensure it's within -180 to 180 degrees
        angle_radians = angle_normalized * np.pi / 180
        total_angle += angle_radians  #DEG_INT * angle * np.pi / 180
        end_x, end_y = xdata[-1] + arm['length'] * np.cos(total_angle), ydata[-1] + arm['length'] * np.sin(total_angle)
        xdata.append(end_x)
        ydata.append(end_y)
    return xdata, ydata

def _config_to_endpoint(config):
    xdata, ydata = _config_to_xy(config)
    return xdata[-1], ydata[-1]









def animate_config(solution):
    
    print('ere')
    fig, ax = plt.subplots()

    total_len = sum([arm['length'] for arm in arm_config])
    ax.set_xlim(-total_len, total_len)
    ax.set_ylim(-total_len, total_len)
    ax.grid(True)
    ax.set_aspect('equal', 'box')

    line, = ax.plot([], [], 'o-', color='blue', label='Current Position')
    
    # Getting start and end endpoints for the arm
    start_x, start_y = _config_to_endpoint(solution[0])
    end_x, end_y = _config_to_endpoint(solution[-1])

    # Adding start and end markers for arm's endpoint
    start_marker, = ax.plot(start_x, start_y, 'go', markersize=8, label='Start Endpoint')
    end_marker, = ax.plot(end_x, end_y, 'ro', markersize=8, label='End Endpoint')
    
    
    # Draw obstacle segments
    for obs in obstacle_segments:
        plt.plot([obs[0].real, obs[1].real], [obs[0].imag, obs[1].imag], 'r-', linewidth=2)


    def init():
        line.set_data([], [])
        return line, start_marker, end_marker

    def update(frame):
        config = solution[frame]
        xdata, ydata = _config_to_xy(config)
        line.set_data(xdata, ydata)
        
        # Ensure start and end markers are plotted at every frame
        start_marker.set_data([start_x], [start_y])
        end_marker.set_data([end_x], [end_y])


        return line, start_marker, end_marker 

    ani = FuncAnimation(fig, update, frames=len(solution), init_func=init, blit=True, repeat=False)
    ax.legend()
    plt.show()
    
    
    
animate_config([(x,x,x) for x in range(360//DEG_INT)])
    
    
import numpy as np
import matplotlib.pyplot as plt

def display_robot_arm(segments):
    fig, ax = plt.subplots()
    for start, end in segments:
        x_values = [start.real, end.real]
        y_values = [start.imag, end.imag]
        ax.plot(x_values, y_values, marker='o')
        
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Robot Arm Visualization')
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    config = [1, 7, 5]  # Replace with actual angles
    setup = {"step_int": 10}  # Replace with the actual step interval
    
    segments = [(0j, (0.766+0.643j)), ((0.766+0.643j), (1.266+1.509j)), ((1.266+1.509j), (1.608+2.449j))]
    display_robot_arm(segments)